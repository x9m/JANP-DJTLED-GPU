/*
MIT License

Copyright (c) 2022 Jinao Zhang
Copyright (c) 2026 Baiqi Cheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// JANP-DJTLED_CPU: OpenMP CPU solver. DJTLED with the average nodal pressure (ANP) formulation for 4-node tetrahedra, NH material.
// The modified volume ratio of an element is computed directly from Jacobian determinants (JANP-DJTLED_Method.pdf, Eq. (9)):
//     J_bar(e) = 1/4 * sum over the 4 nodes a of e: [ sum_{e' at a} det(Jt(e')) / sum_{e' at a} det(J0(e')) ]
// The GPU solver, src/JANP-DJTLED_CUDA.cu, reaches the same J_bar through nodal pressures p_a = Kappa * (J - 1) averaged with
// nodal volumes. The two routes are equal in exact arithmetic for a single material (Kappa cancels, JANP-DJTLED_Method.pdf
// Eqs. (2)-(9)), so the two solvers agree to float rounding. Eq. (10), for models with more than one bulk modulus, is not
// needed: one material per model.
// Build (from the repository root; create bin/ first):
//   Linux, GCC:         g++ -std=c++17 -O3 -fopenmp src/JANP-DJTLED_CPU.cpp -o bin/JANP-DJTLED_CPU
//   macOS, Apple clang: clang++ -std=c++17 -O3 -Xclang -fopenmp -I"$(brew --prefix libomp)/include" src/JANP-DJTLED_CPU.cpp -L"$(brew --prefix libomp)/lib" -lomp -o bin/JANP-DJTLED_CPU
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <cstdio>
#include <string>
#include "vtk_output.h" // VTK export (animation and final shape) and the timers
using namespace  std;
static const int NUM_THREADS(omp_get_max_threads());

// matrix computation/operations (mat: matrix, 33: 3 rows by 3 columns, x: multiplication, T: transpose, Det: determinant, Inv: inverse)
void mat33x33    (const float A[3][3], const float B[3][3], float AB[3][3]);
void mat33Tx33   (const float A[3][3], const float B[3][3], float AB[3][3]);
void mat33x33T   (const float A[3][3], const float B[3][3], float AB[3][3]);
void mat34x34T   (const float A[3][4], const float B[3][4], float AB[3][3]);
void mat33xScalar(const float A[3][3], const float b,       float Ab[3][3]);
void matDet33    (const float A[3][3], float &detA);
void matInv33    (const float A[3][3], float invA[3][3], float &detA);

// classes
class Node;
class T4;
class Model;
class ModelStates;

// methods
Model*       readModel       (int argc, char **argv);
void         printInfo       (const Model& model);
ModelStates* runSimulation   (const Model& model, VtkExport& vtk, size_t frames);
void         initBC          (const Model& model, ModelStates& modelstates);
void         computeRunTimeBC(const Model& model, ModelStates& modelstates, const size_t curr_step);
bool         computeOneStep  (const Model& model, ModelStates& modelstates);
bool         exportVTK       (const Model& model, const ModelStates& modelstates, VtkExport& vtk);
VtkExport    makeVtkExport   (const Model& model);

class Node
{
public:
    const unsigned int m_idx;
    const float        m_x, m_y, m_z;
    Node(const unsigned int idx, const float x, const float y, const float z) :
        m_idx(idx), m_x(x), m_y(y), m_z(z) {};
};

class T4
{
public:
    const unsigned int m_idx, m_n_idx[4];
    const float        m_DHDr[3][4]; //m_DHDrt[4][3];
    float              m_Vol, m_mass,
                       m_J0[3][3], m_detJ0,
                       m_m1[6],
                       m_I1m[3][3],
                       m_twothird_DwDI1cap;
    const string       m_material_type;
    vector<float>      m_material_vals;
    T4(const unsigned int idx, const Node& n1, const Node& n2, const Node& n3, const Node& n4, const float rho, const string material_type, const vector<float>& material_vals) :
        m_idx(idx), m_n_idx{ n1.m_idx, n2.m_idx, n3.m_idx, n4.m_idx },
        m_DHDr{{-1, 1, 0, 0},
               {-1, 0, 1, 0},
               {-1, 0, 0, 1}},
        m_material_type(material_type), m_material_vals(material_vals.cbegin(), material_vals.cend()) // in the order the members are declared

        //m_DHDrt{{-1, -1, -1},
        //         {1, 0, 0},
        //         {0, 1, 0},
        //         {0, 0, 1}}


    {
        float n_coords[3][4], invJ0[3][3], G[6][3][3];
        n_coords[0][0] = n1.m_x; n_coords[1][0] = n1.m_y; n_coords[2][0] = n1.m_z;
        n_coords[0][1] = n2.m_x; n_coords[1][1] = n2.m_y; n_coords[2][1] = n2.m_z;
        n_coords[0][2] = n3.m_x; n_coords[1][2] = n3.m_y; n_coords[2][2] = n3.m_z;
        n_coords[0][3] = n4.m_x; n_coords[1][3] = n4.m_y; n_coords[2][3] = n4.m_z;
        mat34x34T(m_DHDr, n_coords, m_J0);
        matInv33(m_J0, invJ0, m_detJ0);
        m_Vol = m_detJ0 / 6.f;
        m_mass = rho * m_Vol;
        float M[6][3][3] =
        {
           {{1, 0, 0},
            {0, 0, 0},
            {0, 0, 0}},
           {{0, 0, 0},
            {0, 1, 0},
            {0, 0, 0}},
           {{0, 0, 0},
            {0, 0, 0},
            {0, 0, 1}},
           {{0, 1, 0},
            {1, 0, 0},
            {0, 0, 0}},
           {{0, 0, 1},
            {0, 0, 0},
            {1, 0, 0}},
           {{0, 0, 0},
            {0, 0, 1},
            {0, 1, 0}},
        };
        for (size_t i = 0; i < 6; i++)
        {
            float temp[3][3];
            mat33x33(invJ0, M[i], temp);
            mat33x33T(temp, invJ0, G[i]); // Eqs. (B.1-6)
        }
        // precomputation for NH: w = Mu / 2 * (I1cap - 3) + Kappa / 2 * (J - 1)^2
        float DwDI1cap = m_material_vals[0] / 2.f,
              invJ0TinvJ0[3][3];
        mat33Tx33(invJ0, invJ0, invJ0TinvJ0);
        mat33xScalar(invJ0TinvJ0, 2.f * m_Vol, m_I1m); // Eq. (37)
        mat33xScalar(m_I1m, DwDI1cap, m_I1m); // (mu/2)*I1m
        m_twothird_DwDI1cap = -2.f / 3.f * DwDI1cap;
        for (size_t i = 0; i < 6; i++) { m_m1[i] = G[i][0][0] + G[i][1][1] + G[i][2][2]; } // Eq. (C.1)
    };
};

class Model
{
public:
    vector<Node*>        m_nodes;
    vector<T4*>          m_tets;
    size_t               m_num_BCs,    m_num_steps,  m_num_DOFs;
    vector<unsigned int> m_disp_idx_x, m_disp_idx_y, m_disp_idx_z,
                         m_fixP_idx_x, m_fixP_idx_y, m_fixP_idx_z;
    vector<float>        m_disp_mag_x, m_disp_mag_y, m_disp_mag_z,
                         m_grav_f_x,   m_grav_f_y,   m_grav_f_z,
                         m_material_vals;
    float                m_dt, m_total_t, m_alpha, m_rho;
    const string         m_fname;
    string               m_ele_type, m_material_type;
    vector<float>        m_nodal_detJ0; // ANP: sum of det(J0) over the elements attached to each node, denominators of Eq. (9)
    unsigned int         m_node_begin_index, m_ele_begin_index,
                        *m_ele_node_local_idx_pair,
                        *m_tracking_num_eles_i_eles_per_node_j;
    Model(const string fname) :
        m_nodes     (0), m_tets      (0),
        m_num_BCs   (0), m_num_steps (0), m_num_DOFs  (0),
        m_disp_idx_x(0), m_disp_idx_y(0), m_disp_idx_z(0),
        m_fixP_idx_x(0), m_fixP_idx_y(0), m_fixP_idx_z(0),
        m_disp_mag_x(0), m_disp_mag_y(0), m_disp_mag_z(0),
        m_grav_f_x  (0), m_grav_f_y  (0), m_grav_f_z  (0),
        m_material_vals(0), m_dt(0.f), m_total_t(0.f), m_alpha(0.f), m_rho(0.f),
        m_fname(fname), m_ele_type(""), m_material_type(""),
        m_nodal_detJ0(0),
        m_node_begin_index(0), m_ele_begin_index(0),
        m_ele_node_local_idx_pair(nullptr), m_tracking_num_eles_i_eles_per_node_j(nullptr) {};
    ~Model()
    {
        for (Node* node : m_nodes) { delete node; }
        for (T4* tet : m_tets)     { delete tet; }
        delete[] m_ele_node_local_idx_pair;
        delete[] m_tracking_num_eles_i_eles_per_node_j;
    };
    void postCreate()
    {
        // below: provide indexing for nodal states (e.g., individual ele nodal internal forces) to avoid race condition in parallel computing
        m_tracking_num_eles_i_eles_per_node_j = new unsigned int[m_nodes.size() * 2]; memset(m_tracking_num_eles_i_eles_per_node_j, 0, sizeof(unsigned int) * m_nodes.size() * 2);
        vector<vector<unsigned int>> nodes_ele_node_local_idx_pair(m_nodes.size());
        for (T4* tet : m_tets) { for (unsigned int m = 0; m < 4; m++) { nodes_ele_node_local_idx_pair[tet->m_n_idx[m]].push_back(tet->m_idx); nodes_ele_node_local_idx_pair[tet->m_n_idx[m]].push_back(m); } }
        vector<unsigned int> eles_per_node(m_nodes.size(), 0);
        unsigned int length(0);
        for (size_t m = 0; m < m_nodes.size(); m++)
        {
            eles_per_node[m] = (unsigned int)nodes_ele_node_local_idx_pair[m].size() / 2;
            length += eles_per_node[m];
        }
        m_ele_node_local_idx_pair = new unsigned int[length * 2];
        unsigned int* p_ele_node_local_idx_pair = m_ele_node_local_idx_pair, tracking(0);
        for (size_t m = 0; m < m_nodes.size(); m++)
        {
            m_tracking_num_eles_i_eles_per_node_j[m * 2 + 0] = tracking;
            m_tracking_num_eles_i_eles_per_node_j[m * 2 + 1] = eles_per_node[m]; tracking += eles_per_node[m];
            for (size_t n = 0; n < eles_per_node[m]; n++)
            {
                *p_ele_node_local_idx_pair = nodes_ele_node_local_idx_pair[m][n * 2 + 0]; p_ele_node_local_idx_pair++; // tet->m_idx
                *p_ele_node_local_idx_pair = nodes_ele_node_local_idx_pair[m][n * 2 + 1]; p_ele_node_local_idx_pair++; // m
            }
        }
        // ANP: sum of det(J0) over the elements attached to each node (= 6 * total initial volume attached to the node), Eq. (9)
        m_nodal_detJ0.assign(m_nodes.size(), 0.f);
        for (T4* tet : m_tets) { for (unsigned int m = 0; m < 4; m++) { m_nodal_detJ0[tet->m_n_idx[m]] += tet->m_detJ0; } }
    }
};

class ModelStates
{
public:
    vector<float> m_external_F,          m_ele_nodal_internal_F,                       // individual ele nodal internal F to avoid race condition, can be summed to get internal_F for nodes
                  m_disp_mag_t,
                  m_central_diff_const1, m_central_diff_const2, m_central_diff_const3,
                  m_prev_U,              m_curr_U,              m_next_U,
                  m_ele_detJt,           m_nodal_J;                                     // ANP: element det(Jt) and nodal volume ratio J_a
    vector<bool>  m_fixP_flag, m_disp_flag;
    ModelStates(const Model& model) :
        m_external_F         (model.m_num_DOFs,   0.f), m_ele_nodal_internal_F(model.m_tets.size() * 4 * 3, 0.f), // each tet has 4 nodes, each node has 3 (x,y,z) DOFs
        m_disp_mag_t         (model.m_num_DOFs,   0.f),
        m_central_diff_const1(model.m_num_DOFs,   0.f), m_central_diff_const2 (model.m_num_DOFs,            0.f), m_central_diff_const3 (model.m_num_DOFs, 0.f),
        m_prev_U             (model.m_num_DOFs,   0.f), m_curr_U              (model.m_num_DOFs,            0.f), m_next_U              (model.m_num_DOFs, 0.f),
        m_ele_detJt          (model.m_tets.size(),  0.f),
        m_nodal_J            (model.m_nodes.size(), 1.f),
        m_fixP_flag          (model.m_num_DOFs, false),
        m_disp_flag          (model.m_num_DOFs, false)
    {
        vector<float> nodal_mass(model.m_num_DOFs, 0.f);
        for (T4* tet : model.m_tets) { for (size_t m = 0; m < 4; m++) { for (size_t n = 0; n < 3; n++) { nodal_mass[tet->m_n_idx[m] * 3 + n] += tet->m_mass / 4.f; } } }
        for (size_t i = 0; i < model.m_num_DOFs; i++)
        {
            m_central_diff_const1[i] = 1.f / (model.m_alpha * nodal_mass[i] / 2.f / model.m_dt + nodal_mass[i] / model.m_dt / model.m_dt);
            m_central_diff_const2[i] = 2.f * nodal_mass[i] * m_central_diff_const1[i] / model.m_dt / model.m_dt;
            m_central_diff_const3[i] = model.m_alpha * nodal_mass[i] * m_central_diff_const1[i] / 2.f / model.m_dt - m_central_diff_const2[i] / 2.f;
        }
    };
};

int main(int argc, char **argv)
{
    size_t frames(0);
    if (!parseFrames(argc, argv, frames)) { return EXIT_FAILURE; } // optional 2nd argument: number of animation frames (default 100)
    Model* model = readModel(argc, argv);
    if (model != nullptr)
    {
        printInfo(*model);
        VtkExport vtk = makeVtkExport(*model);
        ModelStates* modelstates = runSimulation(*model, vtk, frames);
        const bool saved = modelstates != nullptr && exportVTK(*model, *modelstates, vtk);
        cout << "\tExport time:\t\t" << vtk.ms() << " ms (" << vtk.framesWritten() << " timestep VTK files and the final VTK files; not included in the computation time)" << endl;
        delete model;
        delete modelstates;
        return saved ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    else { return EXIT_FAILURE; }
}

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
Model* readModel(int argc, char **argv)
{
    if (argc - 1 == 0) { cerr << "\n\tError: missing input argument (e.g., NH.txt)." << endl; return nullptr; }
    FILE* file;
    file = fopen(argv[1], "r");
    if (file == nullptr) { cerr << "\n\tError: cannot open file: " << argv[1] << endl; return nullptr; }
    else
    {
        Model* model = new Model(argv[1]);
        char buffer[256];
        unsigned int idx(0); float x(0.f), y(0.f), z(0.f);
        fscanf(file, "%u %f %f %f", &idx, &x, &y, &z); model->m_node_begin_index = idx; model->m_nodes.push_back(new Node(idx - model->m_node_begin_index, x, y, z)); // for first node only, to get node begin index
        while (fscanf(file, "%u %f %f %f", &idx, &x, &y, &z) == 4) { model->m_nodes.push_back(new Node(idx - model->m_node_begin_index, x, y, z)); } // internally, node index starts at 0
        fscanf(file, "%255s", buffer); model->m_material_type = buffer;
        if (model->m_material_type == "NH") { float Mu(0.f), K(0.f); fscanf(file, "%f %f", &Mu, &K); model->m_material_vals.push_back(Mu); model->m_material_vals.push_back(K); }
        else if (model->m_material_type == "other_material_types") { /*add your code here*/ }
        if (model->m_material_vals.size() < 2) { cerr << "\n\tError: unsupported material type: " << model->m_material_type << " (only NH is implemented)." << endl; fclose(file); delete model; return nullptr; }
        fscanf(file, "%255s %f", buffer, &model->m_rho);
        fscanf(file, "%255s", buffer); model->m_ele_type = buffer;
        unsigned int n1_idx(0), n2_idx(0), n3_idx(0), n4_idx(0);
        fscanf(file, "%u %u %u %u %u", &idx, &n1_idx, &n2_idx, &n3_idx, &n4_idx); model->m_ele_begin_index = idx; model->m_tets.push_back(new T4(idx - model->m_ele_begin_index,
                                                                                                                                                   *model->m_nodes[n1_idx - model->m_node_begin_index], *model->m_nodes[n2_idx - model->m_node_begin_index], *model->m_nodes[n3_idx - model->m_node_begin_index], *model->m_nodes[n4_idx - model->m_node_begin_index],
                                                                                                                                                    model->m_rho, model->m_material_type, model->m_material_vals)); // for first ele only, to get ele begin index
        while (fscanf(file, "%u %u %u %u %u", &idx, &n1_idx, &n2_idx, &n3_idx, &n4_idx) == 5) { model->m_tets.push_back(new T4(idx - model->m_ele_begin_index,
                                                                                                                            *model->m_nodes[n1_idx - model->m_node_begin_index], *model->m_nodes[n2_idx - model->m_node_begin_index], *model->m_nodes[n3_idx - model->m_node_begin_index], *model->m_nodes[n4_idx - model->m_node_begin_index],
                                                                                                                             model->m_rho, model->m_material_type, model->m_material_vals)); } // internally, ele index starts at 0
        while (fscanf(file, "%255s", buffer) == 1)
        {
            string BC_type(buffer), xyz("");
            if (BC_type == "<Disp>") // Displacements
            {
                float u(0.f); fscanf(file, "%255s %f", buffer, &u); xyz = buffer;
                if      (xyz == "x") { while (fscanf(file, "%u", &idx) == 1) { model->m_disp_idx_x.push_back(idx - model->m_node_begin_index); model->m_disp_mag_x.push_back(u); } }
                else if (xyz == "y") { while (fscanf(file, "%u", &idx) == 1) { model->m_disp_idx_y.push_back(idx - model->m_node_begin_index); model->m_disp_mag_y.push_back(u); } }
                else if (xyz == "z") { while (fscanf(file, "%u", &idx) == 1) { model->m_disp_idx_z.push_back(idx - model->m_node_begin_index); model->m_disp_mag_z.push_back(u); } }
                model->m_num_BCs++;
            }
            else if (BC_type == "<FixP>") // Fixed positions
            {
                fscanf(file, "%255s", buffer); xyz = buffer;
                if      (xyz == "x")   { while (fscanf(file, "%u", &idx) == 1) { model->m_fixP_idx_x.push_back(idx - model->m_node_begin_index); } }
                else if (xyz == "y")   { while (fscanf(file, "%u", &idx) == 1) { model->m_fixP_idx_y.push_back(idx - model->m_node_begin_index); } }
                else if (xyz == "z")   { while (fscanf(file, "%u", &idx) == 1) { model->m_fixP_idx_z.push_back(idx - model->m_node_begin_index); } }
                else if (xyz == "all") { while (fscanf(file, "%u", &idx) == 1) { model->m_fixP_idx_x.push_back(idx - model->m_node_begin_index); model->m_fixP_idx_y.push_back(idx - model->m_node_begin_index); model->m_fixP_idx_z.push_back(idx - model->m_node_begin_index); } }
                model->m_num_BCs++;
            }
            else if (BC_type == "<Gravity>") // Gravity
            {
                float g(0.f); fscanf(file, "%255s %f", buffer, &g); xyz = buffer;
                if      (xyz == "x") { model->m_grav_f_x.resize(model->m_nodes.size(), 0.f); for (T4* tet : model->m_tets) { for (size_t m = 0; m < 4; m++) { model->m_grav_f_x[tet->m_n_idx[m]] += tet->m_mass * g / 4.f; } } }
                else if (xyz == "y") { model->m_grav_f_y.resize(model->m_nodes.size(), 0.f); for (T4* tet : model->m_tets) { for (size_t m = 0; m < 4; m++) { model->m_grav_f_y[tet->m_n_idx[m]] += tet->m_mass * g / 4.f; } } }
                else if (xyz == "z") { model->m_grav_f_z.resize(model->m_nodes.size(), 0.f); for (T4* tet : model->m_tets) { for (size_t m = 0; m < 4; m++) { model->m_grav_f_z[tet->m_n_idx[m]] += tet->m_mass * g / 4.f; } } }
                model->m_num_BCs++;
            }
            else if (BC_type == "other_BC_types") { /*add your code here*/ }
            else if (BC_type == "</BC>") { break; }
        }
        fscanf(file, "%255s %f", buffer, &model->m_alpha);
        fscanf(file, "%255s %f", buffer, &model->m_dt);
        fscanf(file, "%255s %f", buffer, &model->m_total_t);
        fclose(file);
        if (!(model->m_dt > 0.f) || !(model->m_total_t > 0.f)) { cerr << "\n\tError: TimeStep and TotalTime must be positive (is the model file incomplete?): " << argv[1] << endl; delete model; return nullptr; }
        model->m_num_steps = (size_t)ceil(model->m_total_t / model->m_dt);
        model->m_num_DOFs = model->m_nodes.size() * 3;
        model->postCreate();
        return model;
    }
}

void printInfo(const Model& model)
{
    cout << endl;
    cout << "\t---------------------------------------------------------------------------------------------------" << endl;
    cout << "\t| Open-source (OpenMP) implementation of:                                                         |" << endl;
    cout << "\t|               <A direct Jacobian total Lagrangian explicit dynamics finite element algorithm... |" << endl;
    cout << "\t|                                             for real-time simulation of hyperelastic materials. |" << endl;
    cout << "\t|                                                                              Zhang, J. (2021).  |" << endl;
    cout << "\t|                International Journal for Numerical Methods in Engineering, 122(20), 5744-5772.  |" << endl;
    cout << "\t|                                                                           doi:10.1002/nme.6772> |" << endl;
    cout << "\t|                                                                                  by Jinao Zhang |" << endl;
    cout << "\t---------------------------------------------------------------------------------------------------" << endl;
    cout << "\tModel:\t\t"      << model.m_fname.c_str()         << endl;
    cout << "\tNodes:\t\t"      << model.m_nodes.size()          << " (" << model.m_num_DOFs    << " DOFs)" << endl;
    cout << "\tElements:\t"     << model.m_tets.size()           << " (" << model.m_ele_type.c_str() << ")" << endl;
    cout << "\tFormulation:\t"  << "DJTLED with average nodal pressure (ANP), J_bar from Jacobian determinants (Eq. 9)" << endl;
    cout << "\tEleMaterial:\t"  << model.m_material_type.c_str() << ":"; for (const float val : model.m_material_vals) { cout << " " << val; } cout << "; Density: " << model.m_rho << endl;
    cout << "\tBC:\t\t"         << model.m_num_BCs               << endl;
    cout << "\tDampingCoef.:\t" << model.m_alpha                 << endl;
    cout << "\tTimeStep:\t"     << model.m_dt                    << endl;
    cout << "\tTotalTime:\t"    << model.m_total_t               << endl;
    cout << "\tNumSteps:\t"     << model.m_num_steps             << endl;
    cout << "\n\tNode index starts at " << model.m_node_begin_index << "." << endl;
    cout << "  \tElem index starts at " << model.m_ele_begin_index  << "." << endl;
}

ModelStates* runSimulation(const Model& model, VtkExport& vtk, size_t frames)
{
    ModelStates* modelstates = new ModelStates(model);
    initBC(model, *modelstates);
    size_t progress(0);
    cout << "\n\tusing " << NUM_THREADS << " threads" << endl;
    // animation: the initial state, every schedule.interval steps and the last step, as legacy VTK files listed in a
    // ParaView ".vtk.series" file (a .pvd collection only works with XML .vtu files)
    const FrameSchedule schedule(model.m_num_steps, frames);
    if (schedule.frames > 0)
    {
        cout << "\tsaving an animation frame every " << schedule.interval << " steps in " << vtk.timestepsDir() << "/" << endl;
        if (vtk.beginAnimation()) { vtk.writeFrame(0, 0.f, modelstates->m_curr_U.data()); } // initial (undeformed) state
    }
    cout << "\tcomputing..." << endl;
    Stopwatch compute_timer; // the time-step loop only: paused while VTK files are written
    compute_timer.start();
    for (size_t step = 0; step < model.m_num_steps; step++) // simulation loop
    {
        if ((float)(step + 1) / (float)model.m_num_steps * 100.f >= progress + 10) { progress += 10; cout << "\t\t\t(" << progress << "%)" << endl; }
        computeRunTimeBC(model, *modelstates, step);
        if (!computeOneStep(model, *modelstates)) { delete modelstates; return nullptr; }
        if (schedule.due(step + 1)) // animation frame: the state after step + 1 steps, at time (step + 1) * dt
        {
            compute_timer.stop();
            vtk.writeFrame(step + 1, (float)(step + 1) * model.m_dt, modelstates->m_curr_U.data());
            compute_timer.start();
        }
    }
    compute_timer.stop();
    vtk.endAnimation();
    if (vtk.framesWritten() > 0) { cout << "\tsaved " << vtk.framesWritten() << " animation frames; open " << vtk.seriesFile() << " in ParaView to play them" << endl; }
    cout << "\n\tComputation time:\t" << compute_timer.ms() << " ms (time-step loop only, without VTK export)" << endl;
    return modelstates;
}

void initBC(const Model& model, ModelStates& modelstates)
{
    fill(modelstates.m_external_F.begin(), modelstates.m_external_F.end(), 0.f);
    // BC:Gravity
    for (size_t i = 0; i < model.m_grav_f_x.size(); i++) { modelstates.m_external_F[i * 3 + 0] += model.m_grav_f_x[i]; }
    for (size_t i = 0; i < model.m_grav_f_y.size(); i++) { modelstates.m_external_F[i * 3 + 1] += model.m_grav_f_y[i]; }
    for (size_t i = 0; i < model.m_grav_f_z.size(); i++) { modelstates.m_external_F[i * 3 + 2] += model.m_grav_f_z[i]; }
    // BC:FixP
    for (size_t i = 0; i < model.m_fixP_idx_x.size(); i++) { modelstates.m_fixP_flag[model.m_fixP_idx_x[i] * 3 + 0] = true; }
    for (size_t i = 0; i < model.m_fixP_idx_y.size(); i++) { modelstates.m_fixP_flag[model.m_fixP_idx_y[i] * 3 + 1] = true; }
    for (size_t i = 0; i < model.m_fixP_idx_z.size(); i++) { modelstates.m_fixP_flag[model.m_fixP_idx_z[i] * 3 + 2] = true; }
    // BC:Disp: which DOFs have a prescribed displacement (applied each step in computeRunTimeBC)
    for (size_t i = 0; i < model.m_disp_idx_x.size(); i++) { modelstates.m_disp_flag[model.m_disp_idx_x[i] * 3 + 0] = true; }
    for (size_t i = 0; i < model.m_disp_idx_y.size(); i++) { modelstates.m_disp_flag[model.m_disp_idx_y[i] * 3 + 1] = true; }
    for (size_t i = 0; i < model.m_disp_idx_z.size(); i++) { modelstates.m_disp_flag[model.m_disp_idx_z[i] * 3 + 2] = true; }
}

void computeRunTimeBC(const Model& model, ModelStates& modelstates, const size_t curr_step)
{
    // BC:Disp
    const float n((curr_step + 1) * model.m_dt / model.m_total_t);
    for (size_t i = 0; i < model.m_disp_idx_x.size(); i++) { modelstates.m_disp_mag_t[model.m_disp_idx_x[i] * 3 + 0] = model.m_disp_mag_x[i] * n; }
    for (size_t i = 0; i < model.m_disp_idx_y.size(); i++) { modelstates.m_disp_mag_t[model.m_disp_idx_y[i] * 3 + 1] = model.m_disp_mag_y[i] * n; }
    for (size_t i = 0; i < model.m_disp_idx_z.size(); i++) { modelstates.m_disp_mag_t[model.m_disp_idx_z[i] * 3 + 2] = model.m_disp_mag_z[i] * n; }
}

bool computeOneStep(const Model& model, ModelStates& modelstates)
{
    bool no_err(true);
    const int num_tets((int)model.m_tets.size()), num_nodes((int)model.m_nodes.size());
#pragma omp parallel num_threads(NUM_THREADS)
    {
        // each thread takes every num_threads-th item; the team OpenMP provides can be smaller than NUM_THREADS, so stride by its real size
        const int id = omp_get_thread_num(), num_threads = omp_get_num_threads();
        float u[3][4], Jt[3][3], g[6], invJt[3][3], detJt(0.f), J(0.f), J23(0.f), I1(0.f), const1(0.f), f[3][4]; // X[4][3], Xb[3][3];

        // current Jacobian operator Jt of a tet from the current displacements, Eq. (11)
        auto computeJt = [&](const T4* tet)
        {
            for (size_t m = 0; m < 4; m++) { for (size_t n = 0; n < 3; n++) { u[n][m] = modelstates.m_curr_U[tet->m_n_idx[m] * 3 + n]; } }
            Jt[0][0] = tet->m_J0[0][0] - u[0][0] + u[0][1]; Jt[0][1] = tet->m_J0[0][1] - u[1][0] + u[1][1]; Jt[0][2] = tet->m_J0[0][2] - u[2][0] + u[2][1];
            Jt[1][0] = tet->m_J0[1][0] - u[0][0] + u[0][2]; Jt[1][1] = tet->m_J0[1][1] - u[1][0] + u[1][2]; Jt[1][2] = tet->m_J0[1][2] - u[2][0] + u[2][2];
            Jt[2][0] = tet->m_J0[2][0] - u[0][0] + u[0][3]; Jt[2][1] = tet->m_J0[2][1] - u[1][0] + u[1][3]; Jt[2][2] = tet->m_J0[2][2] - u[2][0] + u[2][3];
        };

        // ANP pass 1: det(Jt) of each element (6 x its current volume, JANP-DJTLED_Method.pdf Eq. (8))
        for (int i = id; i < num_tets; i += num_threads)
        {
            T4 *tet = model.m_tets[i];
            computeJt(tet);
            matDet33(Jt, detJt);
            modelstates.m_ele_detJt[tet->m_idx] = detJt;
        }
#pragma omp barrier
        // ANP pass 2: nodal volume ratio J_a = sum_e det(Jt(e)) / sum_e det(J0(e)) over the elements attached to node a,
        // the bracketed terms of JANP-DJTLED_Method.pdf Eq. (9); no pressures or bulk modulus needed
        // (gathered per node via the node-to-element map, so there is no race condition)
        for (int i = id; i < num_nodes; i += num_threads)
        {
            unsigned int tracking_num_eles(model.m_tracking_num_eles_i_eles_per_node_j[i * 2 + 0]),
                         eles_per_node    (model.m_tracking_num_eles_i_eles_per_node_j[i * 2 + 1]);
            float detJt_sum(0.f);
            for (unsigned int j = 0; j < eles_per_node; j++)
            {
                detJt_sum += modelstates.m_ele_detJt[model.m_ele_node_local_idx_pair[(tracking_num_eles + j) * 2 + 0]];
            }
            modelstates.m_nodal_J[i] = detJt_sum / model.m_nodal_detJ0[i];
        }
#pragma omp barrier

        // ANP pass 3: element forces with modified Jt
        for (int i = id; i < num_tets; i += num_threads) // loop through tets to compute for force contributions
        {
            T4 *tet = model.m_tets[i];
            computeJt(tet); // Eq. (11)
            // modified volume ratio J_bar = average of the 4 nodal volume ratios, JANP-DJTLED_Method.pdf Eq. (9)
            const float J_bar = (modelstates.m_nodal_J[tet->m_n_idx[0]] + modelstates.m_nodal_J[tet->m_n_idx[1]] +
                                 modelstates.m_nodal_J[tet->m_n_idx[2]] + modelstates.m_nodal_J[tet->m_n_idx[3]]) / 4.f;
            // modified Jt = (J_bar / J_e)^(1/3) * Jt with J_e = det(Jt) / det(J0); det(modified Jt) / det(J0) = J_bar
            const float scale = cbrtf(J_bar * tet->m_detJ0 / modelstates.m_ele_detJt[tet->m_idx]);
            for (size_t m = 0; m < 3; m++) { for (size_t n = 0; n < 3; n++) { Jt[m][n] *= scale; } }
            // Eq. (17)
            g[0] = Jt[0][0] * Jt[0][0] + Jt[0][1] * Jt[0][1] + Jt[0][2] * Jt[0][2]; // g11
            g[1] = Jt[1][0] * Jt[1][0] + Jt[1][1] * Jt[1][1] + Jt[1][2] * Jt[1][2]; // g22
            g[2] = Jt[2][0] * Jt[2][0] + Jt[2][1] * Jt[2][1] + Jt[2][2] * Jt[2][2]; // g33
            g[3] = Jt[0][0] * Jt[1][0] + Jt[0][1] * Jt[1][1] + Jt[0][2] * Jt[1][2]; // g12
            g[4] = Jt[0][0] * Jt[2][0] + Jt[0][1] * Jt[2][1] + Jt[0][2] * Jt[2][2]; // g13
            g[5] = Jt[1][0] * Jt[2][0] + Jt[1][1] * Jt[2][1] + Jt[1][2] * Jt[2][2]; // g23
            matInv33(Jt, invJt, detJt);
            J = detJt / tet->m_detJ0; // Eq. (12); with ANP this equals J_bar, since Jt has been modified

            //matInv33(tet->m_DHDrt, invJt, X);
            //mat34x34T(u, X, Xb);
            //J = 1.f + (X[0][0] + X[1][1] + X[2][2]) *0.0000000000001f; // Eq. (13)


            J23 = powf(J, -0.6666667f);
            I1 = tet->m_m1[0] * g[0] + tet->m_m1[1] * g[1] + tet->m_m1[2] * g[2] + tet->m_m1[3] * g[3] + tet->m_m1[4] * g[4] + tet->m_m1[5] * g[5]; // Eq. (27)
            const1 = (tet->m_twothird_DwDI1cap * J23 * I1 + tet->m_material_vals[1] * J * (J - 1.f)) * tet->m_Vol;
            // compute ele f (Table3 NH)
            f[0][1] = J23 * (Jt[0][0] * tet->m_I1m[0][0] + Jt[1][0] * tet->m_I1m[1][0] + Jt[2][0] * tet->m_I1m[2][0]) + const1 * invJt[0][0];
            f[1][1] = J23 * (Jt[0][1] * tet->m_I1m[0][0] + Jt[1][1] * tet->m_I1m[1][0] + Jt[2][1] * tet->m_I1m[2][0]) + const1 * invJt[1][0];
            f[2][1] = J23 * (Jt[0][2] * tet->m_I1m[0][0] + Jt[1][2] * tet->m_I1m[1][0] + Jt[2][2] * tet->m_I1m[2][0]) + const1 * invJt[2][0];
            f[0][2] = J23 * (Jt[0][0] * tet->m_I1m[0][1] + Jt[1][0] * tet->m_I1m[1][1] + Jt[2][0] * tet->m_I1m[2][1]) + const1 * invJt[0][1];
            f[1][2] = J23 * (Jt[0][1] * tet->m_I1m[0][1] + Jt[1][1] * tet->m_I1m[1][1] + Jt[2][1] * tet->m_I1m[2][1]) + const1 * invJt[1][1];
            f[2][2] = J23 * (Jt[0][2] * tet->m_I1m[0][1] + Jt[1][2] * tet->m_I1m[1][1] + Jt[2][2] * tet->m_I1m[2][1]) + const1 * invJt[2][1];
            f[0][3] = J23 * (Jt[0][0] * tet->m_I1m[0][2] + Jt[1][0] * tet->m_I1m[1][2] + Jt[2][0] * tet->m_I1m[2][2]) + const1 * invJt[0][2];
            f[1][3] = J23 * (Jt[0][1] * tet->m_I1m[0][2] + Jt[1][1] * tet->m_I1m[1][2] + Jt[2][1] * tet->m_I1m[2][2]) + const1 * invJt[1][2];
            f[2][3] = J23 * (Jt[0][2] * tet->m_I1m[0][2] + Jt[1][2] * tet->m_I1m[1][2] + Jt[2][2] * tet->m_I1m[2][2]) + const1 * invJt[2][2];
            f[0][0] = -f[0][1] - f[0][2] - f[0][3];
            f[1][0] = -f[1][1] - f[1][2] - f[1][3];
            f[2][0] = -f[2][1] - f[2][2] - f[2][3];
            for (size_t m = 0; m < 4; m++) { for (size_t n = 0; n < 3; n++) { modelstates.m_ele_nodal_internal_F[tet->m_idx * 12 + m * 3 + n] = f[n][m]; } }
        }
#pragma omp barrier
        for (int i = id; i < num_nodes; i += num_threads) // loop through nodes to compute for new displacements U
        {
            // assemble nodal forces from individual ele nodal forces, due to avoiding race condition
            float nodal_internal_F[3] = { 0.f, 0.f, 0.f };
            unsigned int tracking_num_eles(model.m_tracking_num_eles_i_eles_per_node_j[i * 2 + 0]),
                         eles_per_node    (model.m_tracking_num_eles_i_eles_per_node_j[i * 2 + 1]),
                         ele_idx(0), node_local_idx(0);
            for (unsigned int j = 0; j < eles_per_node; j++)
            {
                ele_idx        = model.m_ele_node_local_idx_pair[(tracking_num_eles + j) * 2 + 0];
                node_local_idx = model.m_ele_node_local_idx_pair[(tracking_num_eles + j) * 2 + 1];
                nodal_internal_F[0] += modelstates.m_ele_nodal_internal_F[ele_idx * 12 + node_local_idx * 3 + 0];
                nodal_internal_F[1] += modelstates.m_ele_nodal_internal_F[ele_idx * 12 + node_local_idx * 3 + 1];
                nodal_internal_F[2] += modelstates.m_ele_nodal_internal_F[ele_idx * 12 + node_local_idx * 3 + 2];
            }
            size_t n_DOF(0);
            for (size_t j = 0; j < 3; j++)
            {
                n_DOF = i * 3 + j;
                if (modelstates.m_disp_flag[n_DOF]) { modelstates.m_next_U[n_DOF] = modelstates.m_disp_mag_t[n_DOF]; } // apply BC:Disp
                else if (modelstates.m_fixP_flag[n_DOF] == true) { modelstates.m_next_U[n_DOF] = 0.f; }                        // apply BC:FixP
                else                                                                                                           // explicit central-difference integration
                {
                    modelstates.m_next_U[n_DOF] = modelstates.m_central_diff_const1[n_DOF] * (modelstates.m_external_F[n_DOF] - nodal_internal_F[j]) +
                                                  modelstates.m_central_diff_const2[n_DOF] * modelstates.m_curr_U[n_DOF] +
                                                  modelstates.m_central_diff_const3[n_DOF] * modelstates.m_prev_U[n_DOF];
                    if (std::isnan(modelstates.m_next_U[n_DOF]))
                    {
#pragma omp atomic write
                        no_err = false; // several threads may report divergence at once
                    }
                }
            }
        }
    }
    if (!no_err) { cerr << "\n\tError: solution diverged, simulation aborted. Try a smaller time step." << endl; }
    else { modelstates.m_prev_U.swap(modelstates.m_curr_U); modelstates.m_curr_U.swap(modelstates.m_next_U); }
    return no_err;
}

VtkExport makeVtkExport(const Model& model) // undeformed node coordinates and element connectivity, for VTK output
{
    vector<float> xyz(model.m_nodes.size() * 3);
    for (const Node* node : model.m_nodes) { xyz[node->m_idx * 3 + 0] = node->m_x; xyz[node->m_idx * 3 + 1] = node->m_y; xyz[node->m_idx * 3 + 2] = node->m_z; }
    vector<unsigned int> tets(model.m_tets.size() * 4);
    for (const T4* tet : model.m_tets) { for (unsigned int m = 0; m < 4; m++) { tets[tet->m_idx * 4 + m] = tet->m_n_idx[m]; } }
    return VtkExport(outputPrefix("JANP-DJTLED_CPU", model.m_fname), xyz, tets); // must be this file's name: run.py looks for JANP-DJTLED_CPU_<model>_...
}

bool exportVTK(const Model&, const ModelStates& modelstates, VtkExport& vtk) // final state: deformed and undeformed shapes, both with the final displacements
{
    cout << "\texporting final results..." << endl;
    // <algorithm>_<model>_u.vtk and <algorithm>_<model>_undeformed.vtk
    const bool saved = vtk.writeFile(vtk.prefix() + "_u.vtk",          modelstates.m_curr_U.data(), true,  "displacement") &&
                       vtk.writeFile(vtk.prefix() + "_undeformed.vtk", modelstates.m_curr_U.data(), false, "displacement");
    if (saved) { cout << "\tVTK saved." << endl; }
    return saved;
}

void mat33x33(const float A[3][3], const float B[3][3], float AB[3][3])
{
    memset(AB, 0, sizeof(float) * 3 * 3);
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 3; k++) { AB[i][j] += A[i][k] * B[k][j]; } } }
}
void mat33Tx33(const float A[3][3], const float B[3][3], float AB[3][3])
{
    memset(AB, 0, sizeof(float) * 3 * 3);
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 3; k++) { AB[i][j] += A[k][i] * B[k][j]; } } }
}
void mat33x33T(const float A[3][3], const float B[3][3], float AB[3][3])
{
    memset(AB, 0, sizeof(float) * 3 * 3);
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 3; k++) { AB[i][j] += A[i][k] * B[j][k]; } } }
}
void mat34x34T(const float A[3][4], const float B[3][4], float AB[3][3])
{
    memset(AB, 0, sizeof(float) * 3 * 3);
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 4; k++) { AB[i][j] += A[i][k] * B[j][k]; } } }
}
void mat33xScalar(const float A[3][3], const float b, float Ab[3][3])
{
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { Ab[i][j] = A[i][j] * b; } }
}
void matDet33(const float A[3][3], float &detA)
{
    detA = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[1][0] * (A[0][1] * A[2][2] - A[0][2] * A[2][1]) + A[2][0] * (A[0][1] * A[1][2] - A[0][2] * A[1][1]);
}
void matInv33(const float A[3][3], float invA[3][3], float &detA)
{
    matDet33(A, detA);
    invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / detA; invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / detA; invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / detA;
    invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / detA; invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / detA; invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / detA;
    invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / detA; invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / detA; invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / detA;
}
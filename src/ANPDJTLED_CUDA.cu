/*
MIT License

Copyright (c) 2022 Jinao Zhang

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
// ANPDJTLED_CUDA: CUDA GPU version of src/cpu/nonlocking/ANPDJTLED_OMP.cpp (same algorithm and results, same output files).
// DJTLED with the average nodal pressure (ANP) formulation for 4-node tetrahedra (T4), ported from MechElementT4DJANP (NH material).
// Model reading, setup and VTK export run on the CPU as before; each time step runs on the GPU:
//   1. computeVolumeRatios:           one thread per tetrahedron, element volume ratio J_e
//   2. computeNodalPressures:         one thread per node, nodal pressure from the J_e of its elements
//   3. computeModifiedElementForces:  one thread per tetrahedron, element forces with the ANP-modified Jacobian
//   4. updateNodes:                   one thread per node, sums its element forces, central-difference update, BCs
// The nodes sum their element forces through the same node-to-element map as the OpenMP version, so there are no atomic operations.
// The timer includes writing the timestep VTK files, as in the OpenMP version.
// Build: nvcc -std=c++17 -O3 src/gpu/nonlocking/ANPDJTLED_CUDA.cu -o bin/ANPDJTLED_CUDA
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <string>
#include "../../common/vtk_output.h" // VTK export (animation and final shape) and the timers
using namespace  std;
static const int BLOCK_SIZE(128); // CUDA threads per block

// abort with a readable message if a CUDA call fails
#define CUDA_CHECK(call) do { cudaError_t err_ = (call); if (err_ != cudaSuccess) { \
    cerr << "\n\tError: CUDA error: " << cudaGetErrorString(err_) << " (" << #call << ", line " << __LINE__ << ")" << endl; exit(EXIT_FAILURE); } } while (0)

// matrix computation/operations (mat: matrix, 33: 3 rows by 3 columns, x: multiplication, T: transpose, Det: determinant, Inv: inverse)
__host__ __device__ void mat33x33    (const float A[3][3], const float B[3][3], float AB[3][3]);
__host__ __device__ void mat33Tx33   (const float A[3][3], const float B[3][3], float AB[3][3]);
__host__ __device__ void mat33x33T   (const float A[3][3], const float B[3][3], float AB[3][3]);
__host__ __device__ void mat34x34T   (const float A[3][4], const float B[3][4], float AB[3][3]);
__host__ __device__ void mat33xScalar(const float A[3][3], const float b,       float Ab[3][3]);
__host__ __device__ void matDet33    (const float A[3][3], float &detA);
__host__ __device__ void matInv33    (const float A[3][3], float invA[3][3], float &detA);

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
                       m_G[6][3][3],
                       m_m1[6],
                       m_I1m[3][3],
                       m_twothird_DwDI1cap;
    const string       m_material_type;
    vector<float>      m_material_vals;
    T4(const unsigned int idx, const Node& n1, const Node& n2, const Node& n3, const Node& n4, const float rho, const string material_type, const vector<float>& material_vals) :
        m_idx(idx), m_n_idx{ n1.m_idx, n2.m_idx, n3.m_idx, n4.m_idx }, m_material_type(material_type), m_material_vals(material_vals.cbegin(), material_vals.cend()),
        m_DHDr{{-1, 1, 0, 0},
               {-1, 0, 1, 0},
               {-1, 0, 0, 1}}

        //m_DHDrt{{-1, -1, -1},
        //         {1, 0, 0},
        //         {0, 1, 0},
        //         {0, 0, 1}}


    {
        float n_coords[3][4], invJ0[3][3];
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
            mat33x33T(temp, invJ0, m_G[i]); // Eqs. (B.1-6)
        }
        // precomputation for NH: w = Mu / 2 * (I1cap - 3) + Kappa / 2 * (J - 1)^2
        float DwDI1cap = m_material_vals[0] / 2.f,
              invJ0TinvJ0[3][3];
        mat33Tx33(invJ0, invJ0, invJ0TinvJ0);
        mat33xScalar(invJ0TinvJ0, 2.f * m_Vol, m_I1m); // Eq. (37)
        mat33xScalar(m_I1m, DwDI1cap, m_I1m); // (mu/2)*I1m
        m_twothird_DwDI1cap = -2.f / 3.f * DwDI1cap;
        for (size_t i = 0; i < 6; i++) { m_m1[i] = m_G[i][0][0] + m_G[i][1][1] + m_G[i][2][2]; } // Eq. (C.1)
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
    vector<float>        m_nodal_vol;   // ANP: initial nodal volume, sum of 1/4 of each attached element volume
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
        m_nodal_vol(0),
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
        // ANP: initial nodal volumes, V_a = sum over attached elements of V_e / 4
        m_nodal_vol.assign(m_nodes.size(), 0.f);
        for (T4* tet : m_tets) { for (unsigned int m = 0; m < 4; m++) { m_nodal_vol[tet->m_n_idx[m]] += tet->m_Vol / 4.f; } }
    }
};

class ModelStates
{
public:
    vector<float> m_external_F,          m_ele_nodal_internal_F,                       // individual ele nodal internal F to avoid race condition, can be summed to get internal_F for nodes
                  m_disp_mag_t,
                  m_central_diff_const1, m_central_diff_const2, m_central_diff_const3,
                  m_prev_U,              m_curr_U,              m_next_U,
                  m_ele_J,               m_nodal_p;                                     // ANP: element volume ratio J_e and nodal pressure p_a
    vector<bool>  m_fixP_flag;
    ModelStates(const Model& model) :
        m_external_F         (model.m_num_DOFs,   0.f), m_ele_nodal_internal_F(model.m_tets.size() * 4 * 3, 0.f), // each tet has 4 nodes, each node has 3 (x,y,z) DOFs
        m_disp_mag_t         (model.m_num_DOFs,   0.f),
        m_central_diff_const1(model.m_num_DOFs,   0.f), m_central_diff_const2 (model.m_num_DOFs,            0.f), m_central_diff_const3 (model.m_num_DOFs, 0.f),
        m_prev_U             (model.m_num_DOFs,   0.f), m_curr_U              (model.m_num_DOFs,            0.f), m_next_U              (model.m_num_DOFs, 0.f),
        m_ele_J              (model.m_tets.size(),  1.f),
        m_nodal_p            (model.m_nodes.size(), 0.f),
        m_fixP_flag          (model.m_num_DOFs, false)
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
        while (fscanf(file, "%u %f %f %f", &idx, &x, &y, &z)) { model->m_nodes.push_back(new Node(idx - model->m_node_begin_index, x, y, z)); } // internally, node index starts at 0
        fscanf(file, "%255s", buffer); model->m_material_type = buffer;
        if (model->m_material_type == "NH") { float Mu(0.f), K(0.f); fscanf(file, "%f %f", &Mu, &K); model->m_material_vals.push_back(Mu); model->m_material_vals.push_back(K); }
        else if (model->m_material_type == "other_material_types") { /*add your code here*/ }
        fscanf(file, "%255s %f", buffer, &model->m_rho);
        fscanf(file, "%255s", buffer); model->m_ele_type = buffer;
        unsigned int n1_idx(0), n2_idx(0), n3_idx(0), n4_idx(0);
        fscanf(file, "%u %u %u %u %u", &idx, &n1_idx, &n2_idx, &n3_idx, &n4_idx); model->m_ele_begin_index = idx; model->m_tets.push_back(new T4(idx - model->m_ele_begin_index,
                                                                                                                                                   *model->m_nodes[n1_idx - model->m_node_begin_index], *model->m_nodes[n2_idx - model->m_node_begin_index], *model->m_nodes[n3_idx - model->m_node_begin_index], *model->m_nodes[n4_idx - model->m_node_begin_index],
                                                                                                                                                    model->m_rho, model->m_material_type, model->m_material_vals)); // for first ele only, to get ele begin index
        while (fscanf(file, "%u %u %u %u %u", &idx, &n1_idx, &n2_idx, &n3_idx, &n4_idx)) { model->m_tets.push_back(new T4(idx - model->m_ele_begin_index,
                                                                                                                            *model->m_nodes[n1_idx - model->m_node_begin_index], *model->m_nodes[n2_idx - model->m_node_begin_index], *model->m_nodes[n3_idx - model->m_node_begin_index], *model->m_nodes[n4_idx - model->m_node_begin_index],
                                                                                                                             model->m_rho, model->m_material_type, model->m_material_vals)); } // internally, ele index starts at 0
        while (fscanf(file, "%255s", buffer))
        {
            string BC_type(buffer), xyz("");
            if (BC_type == "<Disp>") // Displacements
            {
                float u(0.f); fscanf(file, "%255s %f", buffer, &u); xyz = buffer;
                if      (xyz == "x") { while (fscanf(file, "%u", &idx)) { model->m_disp_idx_x.push_back(idx - model->m_node_begin_index); model->m_disp_mag_x.push_back(u); } }
                else if (xyz == "y") { while (fscanf(file, "%u", &idx)) { model->m_disp_idx_y.push_back(idx - model->m_node_begin_index); model->m_disp_mag_y.push_back(u); } }
                else if (xyz == "z") { while (fscanf(file, "%u", &idx)) { model->m_disp_idx_z.push_back(idx - model->m_node_begin_index); model->m_disp_mag_z.push_back(u); } }
                model->m_num_BCs++;
            }
            else if (BC_type == "<FixP>") // Fixed positions
            {
                fscanf(file, "%255s", buffer); xyz = buffer;
                if      (xyz == "x")   { while (fscanf(file, "%u", &idx)) { model->m_fixP_idx_x.push_back(idx - model->m_node_begin_index); } }
                else if (xyz == "y")   { while (fscanf(file, "%u", &idx)) { model->m_fixP_idx_y.push_back(idx - model->m_node_begin_index); } }
                else if (xyz == "z")   { while (fscanf(file, "%u", &idx)) { model->m_fixP_idx_z.push_back(idx - model->m_node_begin_index); } }
                else if (xyz == "all") { while (fscanf(file, "%u", &idx)) { model->m_fixP_idx_x.push_back(idx - model->m_node_begin_index); model->m_fixP_idx_y.push_back(idx - model->m_node_begin_index); model->m_fixP_idx_z.push_back(idx - model->m_node_begin_index); } }
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
    cout << "\t| Oepn-source (CUDA) implmentation of:                                                            |" << endl;
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
    cout << "\tFormulation:\t"  << "DJTLED with average nodal pressure (ANP)" << endl;
    cout << "\tEleMaterial:\t"  << model.m_material_type.c_str() << ":"; for (const float val : model.m_material_vals) { cout << " " << val; } cout << "; Density: " << model.m_rho << endl;
    cout << "\tBC:\t\t"         << model.m_num_BCs               << endl;
    cout << "\tDampingCoef.:\t" << model.m_alpha                 << endl;
    cout << "\tTimeStep:\t"     << model.m_dt                    << endl;
    cout << "\tTotalTime:\t"    << model.m_total_t               << endl;
    cout << "\tNumSteps:\t"     << model.m_num_steps             << endl;
    cout << "\n\tNode index starts at " << model.m_node_begin_index << "." << endl;
    cout << "  \tElem index starts at " << model.m_ele_begin_index  << "." << endl;
}


// ---- GPU kernels -------------------------------------------------------------------------------------------------------------------
// Element data on the GPU, flattened per element at position tet->m_idx (the same indexing as the CPU code)
struct EleArrays
{
    const int*   nodes;             // 4 node indices
    const float* J0;                // reference Jacobian operator J0, 9 values (row-major)
    const float* detJ0;             // det(J0)
    const float* I1m;               // (mu/2) * I1m, 9 values (row-major), Eq. (37)
    const float* m1;                // m1, 6 values, Eq. (C.1)
    const float* twothird_DwDI1cap; // -2/3 * DwDI1cap
    const float* K;                 // bulk modulus Kappa
    const float* vol;               // initial volume
};

// current Jacobian operator Jt of element e from the current displacements, Eq. (11)
__device__ void computeJt(const int e, const EleArrays E, const float* __restrict__ curr_U, float Jt[3][3])
{
    float u[3][4];
    for (int m = 0; m < 4; m++) { const int node = E.nodes[e * 4 + m]; for (int n = 0; n < 3; n++) { u[n][m] = curr_U[node * 3 + n]; } }
    const float* J0 = E.J0 + e * 9;
    Jt[0][0] = J0[0] - u[0][0] + u[0][1]; Jt[0][1] = J0[1] - u[1][0] + u[1][1]; Jt[0][2] = J0[2] - u[2][0] + u[2][1];
    Jt[1][0] = J0[3] - u[0][0] + u[0][2]; Jt[1][1] = J0[4] - u[1][0] + u[1][2]; Jt[1][2] = J0[5] - u[2][0] + u[2][2];
    Jt[2][0] = J0[6] - u[0][0] + u[0][3]; Jt[2][1] = J0[7] - u[1][0] + u[1][3]; Jt[2][2] = J0[8] - u[2][0] + u[2][3];
}

// element nodal forces from Jt, NH material (Table 3), stored in the element's own slot of ele_F
__device__ void storeElementForces(const int e, const float Jt[3][3], const EleArrays E, float* __restrict__ ele_F)
{
    float g[6], invJt[3][3], detJt(0.f), f[3][4];
    // Eq. (17)
    g[0] = Jt[0][0] * Jt[0][0] + Jt[0][1] * Jt[0][1] + Jt[0][2] * Jt[0][2]; // g11
    g[1] = Jt[1][0] * Jt[1][0] + Jt[1][1] * Jt[1][1] + Jt[1][2] * Jt[1][2]; // g22
    g[2] = Jt[2][0] * Jt[2][0] + Jt[2][1] * Jt[2][1] + Jt[2][2] * Jt[2][2]; // g33
    g[3] = Jt[0][0] * Jt[1][0] + Jt[0][1] * Jt[1][1] + Jt[0][2] * Jt[1][2]; // g12
    g[4] = Jt[0][0] * Jt[2][0] + Jt[0][1] * Jt[2][1] + Jt[0][2] * Jt[2][2]; // g13
    g[5] = Jt[1][0] * Jt[2][0] + Jt[1][1] * Jt[2][1] + Jt[1][2] * Jt[2][2]; // g23
    matInv33(Jt, invJt, detJt);
    const float J = detJt / E.detJ0[e]; // Eq. (12)
    const float J23 = powf(J, -0.6666667f);
    const float* m1  = E.m1  + e * 6;
    const float* I1m = E.I1m + e * 9;
    const float I1 = m1[0] * g[0] + m1[1] * g[1] + m1[2] * g[2] + m1[3] * g[3] + m1[4] * g[4] + m1[5] * g[5]; // Eq. (27)
    const float const1 = (E.twothird_DwDI1cap[e] * J23 * I1 + E.K[e] * J * (J - 1.f)) * E.vol[e];
    for (int c = 0; c < 3; c++) // nodes 2-4 (Table 3, NH)
    {
        for (int n = 0; n < 3; n++)
        {
            f[n][c + 1] = J23 * (Jt[0][n] * I1m[0 * 3 + c] + Jt[1][n] * I1m[1 * 3 + c] + Jt[2][n] * I1m[2 * 3 + c]) + const1 * invJt[n][c];
        }
    }
    for (int n = 0; n < 3; n++) { f[n][0] = -f[n][1] - f[n][2] - f[n][3]; } // node 1
    for (int m = 0; m < 4; m++) { for (int n = 0; n < 3; n++) { ele_F[e * 12 + m * 3 + n] = f[n][m]; } }
}

// ANP kernel 1, one thread per tetrahedron (MechElementT4DJANP::computePressure): element volume ratio J_e = det(Jt) / det(J0)
__global__ void computeVolumeRatios(const int num_tets, const EleArrays E, const float* __restrict__ curr_U, float* __restrict__ ele_J)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_tets) { return; }
    float Jt[3][3], detJt(0.f);
    computeJt(e, E, curr_U, Jt);
    matDet33(Jt, detJt);
    ele_J[e] = detJt / E.detJ0[e];
}

// ANP kernel 2, one thread per node: nodal pressure p_a = sum_e (p_e * V_e / 4) / V_a, with p_e = Kappa * (J_e - 1), Eqs. (6) and (18)
__global__ void computeNodalPressures(const int num_nodes, const unsigned int* __restrict__ tracking, const unsigned int* __restrict__ pairs,
                                      const EleArrays E, const float* __restrict__ ele_J, const float* __restrict__ nodal_vol, float* __restrict__ nodal_p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) { return; }
    const unsigned int tracking_num_eles(tracking[i * 2 + 0]), eles_per_node(tracking[i * 2 + 1]);
    float p_sum(0.f);
    for (unsigned int j = 0; j < eles_per_node; j++)
    {
        const unsigned int e = pairs[(tracking_num_eles + j) * 2 + 0];
        p_sum += E.K[e] * (ele_J[e] - 1.f) * E.vol[e] / 4.f;
    }
    nodal_p[i] = p_sum / nodal_vol[i];
}

// ANP kernel 3, one thread per tetrahedron (MechElementT4DJANP::computeModifiedInternalForcesNH): element forces with the modified Jt
__global__ void computeModifiedElementForces(const int num_tets, const EleArrays E, const float* __restrict__ curr_U,
                                             const float* __restrict__ ele_J, const float* __restrict__ nodal_p, float* __restrict__ ele_F)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_tets) { return; }
    float Jt[3][3];
    computeJt(e, E, curr_U, Jt); // Eq. (11)
    // average element pressure from its 4 nodes, Eq. (13)
    const float p_AEP = (nodal_p[E.nodes[e * 4 + 0]] + nodal_p[E.nodes[e * 4 + 1]] + nodal_p[E.nodes[e * 4 + 2]] + nodal_p[E.nodes[e * 4 + 3]]) / 4.f;
    // modified volume ratio, Eq. (23)
    const float J_AEP = p_AEP / E.K[e] + 1.f;
    // modified Jt = (J_AEP / J_e)^(1/3) * Jt, Eq. (24); det(modified Jt) / det(J0) = J_AEP
    const float scale = powf(J_AEP, 0.3333333f) * powf(ele_J[e], -0.3333333f);
    for (int m = 0; m < 3; m++) { for (int n = 0; n < 3; n++) { Jt[m][n] *= scale; } }
    storeElementForces(e, Jt, E, ele_F);
}

// Last kernel, one thread per node: sum the node's element forces (same node-to-element map and order as the CPU code), then apply
// BC:Disp, else BC:FixP, else the explicit central-difference update, in the same order as computeOneStep in the CPU code
__global__ void updateNodes(const int num_nodes, const unsigned int* __restrict__ tracking, const unsigned int* __restrict__ pairs,
                            const float* __restrict__ ele_F, const float* __restrict__ external_F,
                            const float* __restrict__ cd_const1, const float* __restrict__ cd_const2, const float* __restrict__ cd_const3,
                            const float* __restrict__ disp_mag, const int* __restrict__ fixP_flag, const float load_factor,
                            const float* __restrict__ prev_U, const float* __restrict__ curr_U, float* __restrict__ next_U, int* __restrict__ diverged)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) { return; }
    float nodal_internal_F[3] = { 0.f, 0.f, 0.f };
    const unsigned int tracking_num_eles(tracking[i * 2 + 0]), eles_per_node(tracking[i * 2 + 1]);
    for (unsigned int j = 0; j < eles_per_node; j++)
    {
        const unsigned int ele_idx = pairs[(tracking_num_eles + j) * 2 + 0], node_local_idx = pairs[(tracking_num_eles + j) * 2 + 1];
        nodal_internal_F[0] += ele_F[ele_idx * 12 + node_local_idx * 3 + 0];
        nodal_internal_F[1] += ele_F[ele_idx * 12 + node_local_idx * 3 + 1];
        nodal_internal_F[2] += ele_F[ele_idx * 12 + node_local_idx * 3 + 2];
    }
    for (int j = 0; j < 3; j++)
    {
        const int n_DOF = i * 3 + j;
        const float disp_t = disp_mag[n_DOF] * load_factor; // BC:Disp with the ramped load curve (0 for DOFs without BC:Disp)
        if (disp_t != 0.f) { next_U[n_DOF] = disp_t; }      // apply BC:Disp
        else if (fixP_flag[n_DOF]) { next_U[n_DOF] = 0.f; } // apply BC:FixP
        else                                                // explicit central-difference integration
        {
            const float u = cd_const1[n_DOF] * (external_F[n_DOF] - nodal_internal_F[j]) + cd_const2[n_DOF] * curr_U[n_DOF] + cd_const3[n_DOF] * prev_U[n_DOF];
            if (isnan(u)) { *diverged = 1; } // checked on the CPU at every export step
            next_U[n_DOF] = u;
        }
    }
}

template <typename T> T* toDevice(const vector<T>& host) // allocate GPU memory and copy a host vector into it
{
    T* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty()) { CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    return dev;
}

ModelStates* runSimulation(const Model& model, VtkExport& vtk, size_t frames)
{
    ModelStates* modelstates = new ModelStates(model);
    initBC(model, *modelstates); // external forces (BC:Gravity) and BC:FixP flags, on the CPU as in the OpenMP version
    size_t progress(0);

    // pick the GPU
    int num_devices(0);
    if (cudaGetDeviceCount(&num_devices) != cudaSuccess || num_devices == 0) { cerr << "\n\tError: no CUDA-capable GPU found, simulation aborted." << endl; delete modelstates; return nullptr; }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "\n\tGPU:\t\t" << prop.name << " (" << prop.multiProcessorCount << " SMs, " << prop.totalGlobalMem / (1024 * 1024) << " MB)" << endl;
    cout << "\tcomputing..." << endl;

    const int num_tets  = (int)model.m_tets.size(),
              num_nodes = (int)model.m_nodes.size(),
              num_DOFs  = (int)model.m_num_DOFs;
    // element data, flattened at position tet->m_idx
    vector<int>   h_nodes(num_tets * 4);
    vector<float> h_J0(num_tets * 9), h_detJ0(num_tets), h_I1m(num_tets * 9), h_m1(num_tets * 6), h_twothird(num_tets), h_K(num_tets), h_vol(num_tets);
    for (const T4* tet : model.m_tets)
    {
        if (tet->m_material_type != "NH") { cerr << "\n\tError: unsupported material type: " << tet->m_material_type.c_str() << ", simulation aborted." << endl; delete modelstates; return nullptr; }
        const unsigned int e = tet->m_idx;
        for (int m = 0; m < 4; m++) { h_nodes[e * 4 + m] = (int)tet->m_n_idx[m]; }
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { h_J0[e * 9 + i * 3 + j] = tet->m_J0[i][j]; h_I1m[e * 9 + i * 3 + j] = tet->m_I1m[i][j]; } }
        for (int i = 0; i < 6; i++) { h_m1[e * 6 + i] = tet->m_m1[i]; }
        h_detJ0[e] = tet->m_detJ0; h_twothird[e] = tet->m_twothird_DwDI1cap; h_K[e] = tet->m_material_vals[1]; h_vol[e] = tet->m_Vol;
    }
    // BC:Disp magnitudes per DOF (where a DOF is listed more than once, the last one wins, as in computeRunTimeBC) and BC:FixP flags
    vector<float> h_disp_mag(num_DOFs, 0.f);
    for (size_t i = 0; i < model.m_disp_idx_x.size(); i++) { h_disp_mag[model.m_disp_idx_x[i] * 3 + 0] = model.m_disp_mag_x[i]; }
    for (size_t i = 0; i < model.m_disp_idx_y.size(); i++) { h_disp_mag[model.m_disp_idx_y[i] * 3 + 1] = model.m_disp_mag_y[i]; }
    for (size_t i = 0; i < model.m_disp_idx_z.size(); i++) { h_disp_mag[model.m_disp_idx_z[i] * 3 + 2] = model.m_disp_mag_z[i]; }
    vector<int> h_fixP(num_DOFs, 0);
    for (int i = 0; i < num_DOFs; i++) { h_fixP[i] = modelstates->m_fixP_flag[i] ? 1 : 0; }
    // node-to-element map built by Model::postCreate
    const vector<unsigned int> h_tracking(model.m_tracking_num_eles_i_eles_per_node_j, model.m_tracking_num_eles_i_eles_per_node_j + num_nodes * 2);
    const size_t num_pairs = num_nodes > 0 ? h_tracking[(num_nodes - 1) * 2 + 0] + h_tracking[(num_nodes - 1) * 2 + 1] : 0;
    const vector<unsigned int> h_pairs(model.m_ele_node_local_idx_pair, model.m_ele_node_local_idx_pair + num_pairs * 2);

    // copy everything to the GPU
    int*          d_nodes      = toDevice(h_nodes);
    float*        d_J0         = toDevice(h_J0);
    float*        d_detJ0      = toDevice(h_detJ0);
    float*        d_I1m        = toDevice(h_I1m);
    float*        d_m1         = toDevice(h_m1);
    float*        d_twothird   = toDevice(h_twothird);
    float*        d_K          = toDevice(h_K);
    float*        d_vol        = toDevice(h_vol);
    unsigned int* d_tracking   = toDevice(h_tracking);
    unsigned int* d_pairs      = toDevice(h_pairs);
    float*        d_external_F = toDevice(modelstates->m_external_F);
    float*        d_cd_const1  = toDevice(modelstates->m_central_diff_const1);
    float*        d_cd_const2  = toDevice(modelstates->m_central_diff_const2);
    float*        d_cd_const3  = toDevice(modelstates->m_central_diff_const3);
    float*        d_disp_mag   = toDevice(h_disp_mag);
    int*          d_fixP       = toDevice(h_fixP);
    float*        d_prev_U     = toDevice(modelstates->m_prev_U); // all zeros at the start
    float*        d_curr_U     = toDevice(modelstates->m_curr_U);
    float*        d_next_U     = toDevice(modelstates->m_next_U);
    float*        d_ele_F      = toDevice(vector<float>((size_t)num_tets * 12, 0.f));
    int*          d_diverged   = toDevice(vector<int>(1, 0));
    float*        d_nodal_vol  = toDevice(model.m_nodal_vol);              // ANP: initial nodal volumes
    float*        d_ele_J      = toDevice(vector<float>(num_tets, 1.f));     // ANP: element volume ratios J_e
    float*        d_nodal_p    = toDevice(vector<float>(num_nodes, 0.f));    // ANP: nodal pressures
    const EleArrays E = { d_nodes, d_J0, d_detJ0, d_I1m, d_m1, d_twothird, d_K, d_vol };
    auto freeAll = [&]()
    {
        for (void* p : { (void*)d_nodes, (void*)d_J0, (void*)d_detJ0, (void*)d_I1m, (void*)d_m1, (void*)d_twothird, (void*)d_K, (void*)d_vol,
                         (void*)d_tracking, (void*)d_pairs, (void*)d_external_F, (void*)d_cd_const1, (void*)d_cd_const2, (void*)d_cd_const3,
                         (void*)d_disp_mag, (void*)d_fixP, (void*)d_prev_U, (void*)d_curr_U, (void*)d_next_U, (void*)d_ele_F, (void*)d_diverged,
                         (void*)d_nodal_vol, (void*)d_ele_J, (void*)d_nodal_p }) { cudaFree(p); }
    };
    const int ele_blocks  = (num_tets  + BLOCK_SIZE - 1) / BLOCK_SIZE,
              node_blocks = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // The timer starts only now, after the one-off GPU setup above (starting CUDA, allocating GPU memory, copying the model to it);
    // the GPU is idle here because those copies are synchronous.
    // Animation: the initial state, every schedule.interval steps and the last step, as legacy VTK files listed in a
    // ParaView ".vtk.series" file (a .pvd collection only works with XML .vtu files)
    const FrameSchedule schedule(model.m_num_steps, frames);
    if (schedule.frames > 0)
    {
        cout << "\tsaving an animation frame every " << schedule.interval << " steps in " << vtk.timestepsDir() << "/" << endl;
        if (vtk.beginAnimation()) { vtk.writeFrame(0, 0.f, modelstates->m_curr_U.data()); } // initial (undeformed) state
    }
    const size_t check_interval = max(size_t(1), model.m_num_steps / 100); // check for CUDA errors and divergence every 1% of the steps
    Stopwatch compute_timer; // the time-step loop only: paused while U is copied back and VTK files are written
    compute_timer.start();
    for (size_t step = 0; step < model.m_num_steps; step++) // simulation loop
    {
        if ((float)(step + 1) / (float)model.m_num_steps * 100.f >= progress + 10) { progress += 10; cout << "\t\t\t(" << progress << "%)" << endl; }
        const float n((step + 1) * model.m_dt / model.m_total_t); // ramped load curve for BC:Disp, as in computeRunTimeBC
        computeVolumeRatios         <<<ele_blocks,  BLOCK_SIZE>>>(num_tets, E, d_curr_U, d_ele_J);
        computeNodalPressures       <<<node_blocks, BLOCK_SIZE>>>(num_nodes, d_tracking, d_pairs, E, d_ele_J, d_nodal_vol, d_nodal_p);
        computeModifiedElementForces<<<ele_blocks,  BLOCK_SIZE>>>(num_tets, E, d_curr_U, d_ele_J, d_nodal_p, d_ele_F);
        updateNodes<<<node_blocks, BLOCK_SIZE>>>(num_nodes, d_tracking, d_pairs, d_ele_F, d_external_F, d_cd_const1, d_cd_const2, d_cd_const3,
                                                 d_disp_mag, d_fixP, n, d_prev_U, d_curr_U, d_next_U, d_diverged);
        // rotate U: curr_U becomes prev_U, next_U becomes curr_U (pointer swap, no copying)
        float* recycled = d_prev_U; d_prev_U = d_curr_U; d_curr_U = d_next_U; d_next_U = recycled;

        // every 1% of the steps and at every animation frame: check for errors and divergence (this also waits for the GPU),
        // then save the frame: the state after step + 1 steps, at time (step + 1) * dt
        if ((step + 1) % check_interval == 0 || schedule.due(step + 1) || step == model.m_num_steps - 1)
        {
            CUDA_CHECK(cudaGetLastError());
            int diverged(0);
            CUDA_CHECK(cudaMemcpy(&diverged, d_diverged, sizeof(int), cudaMemcpyDeviceToHost));
            if (diverged) { cerr << "\n\tError: solution diverged, simulation aborted. Try a smaller time step." << endl; freeAll(); delete modelstates; return nullptr; }
            if (schedule.due(step + 1))
            {
                compute_timer.stop();
                vtk.timer().start(); // copying U back from the GPU counts as export time
                CUDA_CHECK(cudaMemcpy(modelstates->m_curr_U.data(), d_curr_U, num_DOFs * sizeof(float), cudaMemcpyDeviceToHost));
                vtk.writeFrame(step + 1, (float)(step + 1) * model.m_dt, modelstates->m_curr_U.data());
                vtk.timer().stop();
                compute_timer.start();
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for the last GPU work before stopping the timer
    compute_timer.stop();
    CUDA_CHECK(cudaGetLastError());
    vtk.endAnimation();
    // final displacements (the last frame copied them too)
    CUDA_CHECK(cudaMemcpy(modelstates->m_curr_U.data(), d_curr_U, num_DOFs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(modelstates->m_prev_U.data(), d_prev_U, num_DOFs * sizeof(float), cudaMemcpyDeviceToHost));
    freeAll();
    if (vtk.framesWritten() > 0) { cout << "\tsaved " << vtk.framesWritten() << " animation frames; open " << vtk.seriesFile() << " in ParaView to play them" << endl; }
    cout << "\n\tComputation time:\t" << compute_timer.ms() << " ms (time-step loop only, without copying U back for, or writing, VTK files)" << endl;
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
}

VtkExport makeVtkExport(const Model& model) // undeformed node coordinates and element connectivity, for VTK output
{
    vector<float> xyz(model.m_nodes.size() * 3);
    for (const Node* node : model.m_nodes) { xyz[node->m_idx * 3 + 0] = node->m_x; xyz[node->m_idx * 3 + 1] = node->m_y; xyz[node->m_idx * 3 + 2] = node->m_z; }
    vector<unsigned int> tets(model.m_tets.size() * 4);
    for (const T4* tet : model.m_tets) { for (unsigned int m = 0; m < 4; m++) { tets[tet->m_idx * 4 + m] = tet->m_n_idx[m]; } }
    return VtkExport("ANPDJTLED_CUDA", xyz, tets);
}

bool exportVTK(const Model& model, const ModelStates& modelstates, VtkExport& vtk) // final state: deformed and undeformed shapes, both with the final displacements
{
    cout << "\texporting final results..." << endl;
    const bool saved = vtk.writeFile("ANPDJTLED_CUDA_brain_everyu.vtk", modelstates.m_curr_U.data(), true,  "displacement") &&
                       vtk.writeFile("ANPDJTLED_CUDA_undeformed.vtk",   modelstates.m_curr_U.data(), false, "displacement");
    if (saved) { cout << "\tVTK saved." << endl; }
    return saved;
}

__host__ __device__ void mat33x33(const float A[3][3], const float B[3][3], float AB[3][3])
{
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { AB[i][j] = 0.f; } }
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 3; k++) { AB[i][j] += A[i][k] * B[k][j]; } } }
}
__host__ __device__ void mat33Tx33(const float A[3][3], const float B[3][3], float AB[3][3])
{
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { AB[i][j] = 0.f; } }
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 3; k++) { AB[i][j] += A[k][i] * B[k][j]; } } }
}
__host__ __device__ void mat33x33T(const float A[3][3], const float B[3][3], float AB[3][3])
{
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { AB[i][j] = 0.f; } }
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 3; k++) { AB[i][j] += A[i][k] * B[j][k]; } } }
}
__host__ __device__ void mat34x34T(const float A[3][4], const float B[3][4], float AB[3][3])
{
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { AB[i][j] = 0.f; } }
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { for (size_t k = 0; k < 4; k++) { AB[i][j] += A[i][k] * B[j][k]; } } }
}
__host__ __device__ void mat33xScalar(const float A[3][3], const float b, float Ab[3][3])
{
    for (size_t i = 0; i < 3; i++) { for (size_t j = 0; j < 3; j++) { Ab[i][j] = A[i][j] * b; } }
}
__host__ __device__ void matDet33(const float A[3][3], float &detA)
{
    detA = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[1][0] * (A[0][1] * A[2][2] - A[0][2] * A[2][1]) + A[2][0] * (A[0][1] * A[1][2] - A[0][2] * A[1][1]);
}
__host__ __device__ void matInv33(const float A[3][3], float invA[3][3], float &detA)
{
    matDet33(A, detA);
    invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / detA; invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / detA; invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / detA;
    invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / detA; invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / detA; invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / detA;
    invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / detA; invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / detA; invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / detA;
}
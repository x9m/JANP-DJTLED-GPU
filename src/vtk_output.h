// vtk_output.h: VTK export and timing shared by the solvers in src/ (header only; each program includes it).
//
//   Stopwatch    wall-clock timer that can be paused, so the simulation time can leave out the VTK export.
//   parseFrames  reads the optional second command-line argument: the number of animation frames (default 100).
//   FrameSchedule decides after which time steps a frame is written: the initial state, every `interval` steps, and the last step.
//   outputPrefix builds the "<algorithm>_<model>" prefix that every output file name starts with (e.g. ANPDJTLED_DET_OMP_NH).
//   VtkExport    writes legacy ASCII VTK files:
//                  - an animation: <prefix>_timesteps/<prefix>_step_NNNNNN.vtk (NNNNNN = time steps completed) and the
//                    ParaView index <prefix>_simulation.vtk.series, which opens as one animated dataset;
//                  - any other file, such as the final shape <prefix>_u.vtk and the undeformed shape <prefix>_undeformed.vtk.
//                Each file is built in memory and written with one call, and the cell block (the same in every file) is
//                formatted once. Numbers are formatted as std::ostream does by default (%g, 6 significant digits), so the
//                text is the same as the previous `fout << value << endl` code. All time spent here is added to ms().
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// Wall-clock stopwatch. start()/stop() may be nested: it runs while at least one start() has no matching stop().
class Stopwatch
{
public:
    void start() { if (m_depth++ == 0) { m_t0 = std::chrono::steady_clock::now(); } }
    void stop()  { if (m_depth > 0 && --m_depth == 0) { m_total += std::chrono::steady_clock::now() - m_t0; } }
    double ms() const
    {
        auto total = m_total;
        if (m_depth > 0) { total += std::chrono::steady_clock::now() - m_t0; }
        return std::chrono::duration<double, std::milli>(total).count();
    }
private:
    std::chrono::steady_clock::duration   m_total{ 0 };
    std::chrono::steady_clock::time_point m_t0;
    int                                   m_depth = 0;
};

// Optional second argument: number of animation frames (default 100; 0 = no animation; >= number of steps = every step).
inline bool parseFrames(int argc, char** argv, size_t& frames)
{
    frames = 100;
    if (argc < 3) { return true; }
    char* end = nullptr;
    const long long value = std::strtoll(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value < 0)
    {
        std::cerr << "\n\tError: the second argument must be the number of animation frames (a whole number, 0 for none), not \"" << argv[2] << "\"." << std::endl;
        return false;
    }
    frames = (size_t)value;
    return true;
}

// Prefix of every output file: "<algorithm>_<model>", where <model> is the model file name without its folder and
// extension (models/NH.txt -> NH). The final files are <prefix>_u.vtk and <prefix>_undeformed.vtk.
inline std::string outputPrefix(const std::string& algorithm, const std::string& model_path)
{
    const std::string model = std::filesystem::path(model_path).stem().string();
    return algorithm + "_" + (model.empty() ? "model" : model);
}

// A frame is written for the initial state (0 steps done), after every `interval` steps, and after the last step.
struct FrameSchedule
{
    size_t num_steps, frames, interval;
    FrameSchedule(size_t num_steps_, size_t frames_) :
        num_steps(num_steps_), frames(frames_), interval(frames_ == 0 ? 0 : std::max<size_t>(1, num_steps_ / frames_)) {}
    bool due(size_t steps_done) const { return frames > 0 && (steps_done % interval == 0 || steps_done == num_steps); }
};

class VtkExport
{
public:
    // xyz: undeformed node coordinates (3 per node, by node index); tets: node indices (4 per tetrahedron, by element index)
    VtkExport(const std::string& prefix, const std::vector<float>& xyz, const std::vector<unsigned int>& tets) :
        m_prefix(prefix), m_xyz(xyz), m_num_nodes(xyz.size() / 3)
    {
        const size_t num_tets = tets.size() / 4;
        char buf[96];
        m_cells.reserve(num_tets * 32);
        std::snprintf(buf, sizeof(buf), "CELLS %zu %zu\n", num_tets, num_tets * (4 + 1)); m_cells += buf;
        for (size_t e = 0; e < num_tets; e++)
        {
            std::snprintf(buf, sizeof(buf), "4 %u %u %u %u\n", tets[e * 4 + 0], tets[e * 4 + 1], tets[e * 4 + 2], tets[e * 4 + 3]);
            m_cells += buf;
        }
        std::snprintf(buf, sizeof(buf), "CELL_TYPES %zu\n", num_tets); m_cells += buf;
        for (size_t e = 0; e < num_tets; e++) { m_cells += "10\n"; } // VTK_TETRA
    }
    ~VtkExport() { endAnimation(); }
    VtkExport(const VtkExport&) = delete;            // owns the open .vtk.series file
    VtkExport& operator=(const VtkExport&) = delete;

    const std::string& prefix() const { return m_prefix; }
    std::string timestepsDir() const { return m_prefix + "_timesteps"; }
    std::string seriesFile()   const { return m_prefix + "_simulation.vtk.series"; }
    size_t framesWritten()     const { return m_frames; }
    double ms()                const { return m_timer.ms(); } // all time spent exporting, including what callers add with timer()
    Stopwatch& timer()               { return m_timer; }      // for callers to add related work (e.g. copying U from the GPU)

    // creates <prefix>_timesteps/ and starts <prefix>_simulation.vtk.series
    bool beginAnimation()
    {
        m_timer.start();
        std::error_code ec;
        std::filesystem::create_directories(timestepsDir(), ec);
        m_series = ec ? nullptr : std::fopen(seriesFile().c_str(), "wb");
        if (m_series) { std::fputs("{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n", m_series); }
        else { std::cerr << "\n\tError: cannot create " << timestepsDir() << "/ or " << seriesFile() << ", animation not saved." << std::endl; }
        m_timer.stop();
        return m_series != nullptr;
    }

    // one animation frame: the deformed shape and displacements after `steps_done` time steps, at simulation time `time`
    bool writeFrame(size_t steps_done, float time, const float* U)
    {
        if (!m_series) { return false; }
        m_timer.start();
        char name[512];
        std::snprintf(name, sizeof(name), "%s/%s_step_%06zu.vtk", timestepsDir().c_str(), m_prefix.c_str(), steps_done);
        const bool ok = writeFile(name, U, true, "displacement", false);
        if (ok)
        {
            char entry[640];
            std::snprintf(entry, sizeof(entry), "%s    { \"name\" : \"%s\", \"time\" : %g }", m_frames == 0 ? "" : ",\n", name, (double)time);
            std::fputs(entry, m_series);
            m_frames++;
        }
        m_timer.stop();
        return ok;
    }

    // closes the .vtk.series file (also done by the destructor, so an aborted run still leaves a valid file)
    void endAnimation()
    {
        if (!m_series) { return; }
        m_timer.start();
        std::fputs("\n  ]\n}\n", m_series);
        std::fclose(m_series);
        m_series = nullptr;
        m_timer.stop();
    }

    // one VTK file. deformed: POINTS are the undeformed coordinates plus U, otherwise the undeformed coordinates.
    // The title line is the file path; the displacement vectors are named vectors_name.
    bool writeFile(const std::string& path, const float* U, bool deformed, const std::string& vectors_name, bool print = true)
    {
        m_timer.start();
        std::string text;
        text.reserve(m_cells.size() + m_num_nodes * 80 + 256);
        char buf[128];
        text += "# vtk DataFile Version 3.8\n";
        text += path; text += "\nASCII\nDATASET UNSTRUCTURED_GRID\n";
        std::snprintf(buf, sizeof(buf), "POINTS %zu float\n", m_num_nodes); text += buf;
        for (size_t i = 0; i < m_num_nodes; i++)
        {
            float x = m_xyz[i * 3 + 0], y = m_xyz[i * 3 + 1], z = m_xyz[i * 3 + 2];
            if (deformed) { x += U[i * 3 + 0]; y += U[i * 3 + 1]; z += U[i * 3 + 2]; } // same float sums as before
            std::snprintf(buf, sizeof(buf), "%g %g %g\n", (double)x, (double)y, (double)z); text += buf;
        }
        text += m_cells;
        std::snprintf(buf, sizeof(buf), "POINT_DATA %zu\n", m_num_nodes); text += buf;
        text += "VECTORS "; text += vectors_name; text += " float\n";
        for (size_t i = 0; i < m_num_nodes; i++)
        {
            std::snprintf(buf, sizeof(buf), "%g %g %g\n", (double)U[i * 3 + 0], (double)U[i * 3 + 1], (double)U[i * 3 + 2]); text += buf;
        }
        FILE* f = std::fopen(path.c_str(), "wb");
        const bool ok = f && std::fwrite(text.data(), 1, text.size(), f) == text.size();
        if (f) { std::fclose(f); }
        m_timer.stop();
        if (!ok) { std::cerr << "\n\tError: cannot write " << path << ", results not saved." << std::endl; }
        else if (print) { std::cout << "\t\t\t" << path << std::endl; }
        return ok;
    }

private:
    std::string        m_prefix, m_cells;
    std::vector<float> m_xyz;
    size_t             m_num_nodes, m_frames = 0;
    FILE*              m_series = nullptr;
    Stopwatch          m_timer;
};

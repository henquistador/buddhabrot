// Offline complex Hénon escape cloud -> standard 3DGS PLY.
//
// Iteration happens in C^2:
//   (z, w) <- (z^2 + c - a*w, z)
// with fixed complex coupling a. Every orbit point has four real coordinates.
// A fixed oblique projection maps that genuine C^2 state into XYZ; no axis is
// orbit time, and no 2D image planes are copied or revolved.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double C_REAL_MIN = -2.15;
constexpr double C_REAL_MAX = 1.15;
constexpr double C_IMAG_MIN = -1.65;
constexpr double C_IMAG_MAX = 1.65;
constexpr double FIELD_MIN = -2.35;
constexpr double FIELD_MAX = 2.35;
constexpr double COUPLING_MAGNITUDE = 0.22;
constexpr double COUPLING_PHASE = 0.65;
constexpr double COUPLING_REAL = 0.1751384374673037;
constexpr double COUPLING_IMAG = 0.1331410089695180;
constexpr float SH_C0 = 0.28209479177387814f;

struct Complex {
  double real = 0.0;
  double imag = 0.0;
};

struct Point3 {
  double x;
  double y;
  double z;
};

struct Options {
  uint64_t samples = 12'000'000;
  uint32_t iterations = 512;
  uint32_t resolution = 864;
  uint32_t min_escape = 8;
  uint32_t max_splats = 1'000'000;
  uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
  std::filesystem::path output = "outputs/buddhabrot/splat.ply";
  std::filesystem::path stats = "public/buddhabrot.json";
};

struct VoxelCount {
  uint32_t voxel;
  uint32_t count;
};

struct Candidate {
  uint32_t voxel;
  float red;
  float green;
  float blue;
  float density;
  double selection_key;
};

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double random01(uint64_t sample, uint64_t lane) {
  const uint64_t bits = splitmix64(sample * 2 + lane + 0x68656e6f6eULL);
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

Complex square(const Complex& value) {
  return {
      value.real * value.real - value.imag * value.imag,
      2.0 * value.real * value.imag,
  };
}

Complex multiply_by_coupling(const Complex& value) {
  return {
      COUPLING_REAL * value.real - COUPLING_IMAG * value.imag,
      COUPLING_REAL * value.imag + COUPLING_IMAG * value.real,
  };
}

void iterate(Complex& z, Complex& w, const Complex& c) {
  const Complex previous = z;
  const Complex squared = square(z);
  const Complex coupled = multiply_by_coupling(w);
  z = {squared.real + c.real - coupled.real,
       squared.imag + c.imag - coupled.imag};
  w = previous;
}

double norm_squared(const Complex& value) {
  return value.real * value.real + value.imag * value.imag;
}

uint32_t escape_time(const Complex& c, uint32_t max_iterations) {
  Complex z;
  Complex w;
  for (uint32_t step = 0; step < max_iterations; ++step) {
    iterate(z, w, c);
    if (norm_squared(z) + 0.35 * norm_squared(w) > 36.0) return step + 1;
  }
  return 0;
}

Point3 project_c2(const Complex& z, const Complex& w) {
  // An oblique 4D camera. The three rows are deliberately mixed so neither
  // z nor w becomes a flat screen plane. Scale balances this Hénon family.
  return {
      0.74 * z.real + 0.26 * w.real + 0.31 * w.imag,
      0.74 * z.imag + 0.26 * w.imag - 0.31 * w.real,
      0.58 * w.real - 0.27 * z.real + 0.46 * w.imag - 0.21 * z.imag,
  };
}

void add_orbit(std::vector<uint32_t>& hits, const Options& options,
               const Complex& c, uint32_t escape) {
  Complex z;
  Complex w;
  const double scale = options.resolution / (FIELD_MAX - FIELD_MIN);
  const size_t plane = static_cast<size_t>(options.resolution) * options.resolution;

  for (uint32_t step = 0; step < escape; ++step) {
    iterate(z, w, c);
    if (step < 2) continue;
    const Point3 point = project_c2(z, w);
    if (point.x < FIELD_MIN || point.x >= FIELD_MAX ||
        point.y < FIELD_MIN || point.y >= FIELD_MAX ||
        point.z < FIELD_MIN || point.z >= FIELD_MAX) continue;

    const uint32_t x = static_cast<uint32_t>((point.x - FIELD_MIN) * scale);
    const uint32_t y = static_cast<uint32_t>((point.y - FIELD_MIN) * scale);
    const uint32_t z_index = static_cast<uint32_t>((point.z - FIELD_MIN) * scale);
    if (x >= options.resolution || y >= options.resolution || z_index >= options.resolution) continue;
    const size_t voxel = static_cast<size_t>(z_index) * plane +
                         static_cast<size_t>(y) * options.resolution + x;
    hits.push_back(static_cast<uint32_t>(voxel));
  }
}

double percentile_log(const std::vector<VoxelCount>& density, double quantile) {
  std::vector<float> values;
  values.reserve(density.size());
  for (const VoxelCount& voxel : density) {
    values.push_back(std::log1p(static_cast<float>(voxel.count)));
  }
  if (values.empty()) return 1.0;
  const size_t index = std::min(values.size() - 1,
      static_cast<size_t>(quantile * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return std::max(1e-6f, values[index]);
}

float normalized_density(uint32_t count, double exposure) {
  const double value = std::min(1.0, std::log1p(static_cast<double>(count)) / exposure);
  return static_cast<float>(std::pow(value, 1.6));
}

void append_float(std::ofstream& output, float value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_ply(const Options& options, const std::vector<Candidate>& splats) {
  std::filesystem::create_directories(options.output.parent_path());
  std::ofstream output(options.output, std::ios::binary);
  if (!output) throw std::runtime_error("could not open output PLY");

  output << "ply\nformat binary_little_endian 1.0\n"
         << "comment offline complex Henon C2 escape-orbit cloud\n"
         << "element vertex " << splats.size() << "\n";
  const char* fields[] = {
      "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
      "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  for (const char* field : fields) output << "property float " << field << "\n";
  output << "end_header\n";

  const size_t plane = static_cast<size_t>(options.resolution) * options.resolution;
  const float sigma = static_cast<float>((FIELD_MAX - FIELD_MIN) / options.resolution * 0.24);
  const float log_sigma = std::log(sigma);

  for (const Candidate& splat : splats) {
    const uint32_t z_index = splat.voxel / plane;
    const uint32_t remainder = splat.voxel % plane;
    const uint32_t y_index = remainder / options.resolution;
    const uint32_t x_index = remainder % options.resolution;
    const auto coordinate = [&](uint32_t index) {
      return static_cast<float>(FIELD_MIN + (index + 0.5) / options.resolution *
                                             (FIELD_MAX - FIELD_MIN));
    };
    const float alpha = std::clamp(0.035f + 0.52f * std::pow(splat.density, 1.1f), 0.02f, 0.555f);
    const float opacity = std::log(alpha / (1.0f - alpha));
    const float values[] = {
        coordinate(x_index), coordinate(y_index), coordinate(z_index),
        0.0f, 0.0f, 0.0f,
        (splat.red - 0.5f) / SH_C0,
        (splat.green - 0.5f) / SH_C0,
        (splat.blue - 0.5f) / SH_C0,
        opacity, log_sigma, log_sigma, log_sigma,
        1.0f, 0.0f, 0.0f, 0.0f};
    for (float value : values) append_float(output, value);
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + arg);
      return argv[i];
    };
    if (arg == "--samples") options.samples = std::stoull(next());
    else if (arg == "--iterations") options.iterations = std::stoul(next());
    else if (arg == "--resolution") options.resolution = std::stoul(next());
    else if (arg == "--min-escape") options.min_escape = std::stoul(next());
    else if (arg == "--max-splats") options.max_splats = std::stoul(next());
    else if (arg == "--threads") options.threads = std::max(1ul, std::stoul(next()));
    else if (arg == "--output") options.output = next();
    else if (arg == "--stats") options.stats = next();
    else throw std::runtime_error("unknown argument: " + arg);
  }
  const uint64_t voxels = static_cast<uint64_t>(options.resolution) *
                          options.resolution * options.resolution;
  if (options.resolution == 0 || voxels > UINT32_MAX) {
    throw std::runtime_error("resolution must fit a 32-bit sparse voxel index");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::atomic<uint64_t> cursor{0};
    std::atomic<uint64_t> escaped{0};
    std::vector<std::vector<uint32_t>> local_hits(options.threads);
    std::vector<std::thread> workers;
    workers.reserve(options.threads);

    std::cerr << "sampling " << options.samples << " complex parameters for H(z,w)="
              << "(z^2+c-a*w,z), |a|=" << COUPLING_MAGNITUDE << ", phase="
              << COUPLING_PHASE << ", " << options.iterations << " iterations into "
              << options.resolution << "^3 voxels\n";

    for (uint32_t thread = 0; thread < options.threads; ++thread) {
      local_hits[thread].reserve(options.samples / options.threads);
      workers.emplace_back([&, thread] {
        constexpr uint64_t CHUNK = 128;
        while (true) {
          const uint64_t begin = cursor.fetch_add(CHUNK);
          if (begin >= options.samples) break;
          const uint64_t end = std::min(options.samples, begin + CHUNK);
          for (uint64_t sample = begin; sample < end; ++sample) {
            const Complex c{
                C_REAL_MIN + random01(sample, 0) * (C_REAL_MAX - C_REAL_MIN),
                C_IMAG_MIN + random01(sample, 1) * (C_IMAG_MAX - C_IMAG_MIN),
            };
            const uint32_t escape = escape_time(c, options.iterations);
            if (escape >= options.min_escape) {
              ++escaped;
              add_orbit(local_hits[thread], options, c, escape);
            }
          }
        }
      });
    }
    for (auto& worker : workers) worker.join();

    size_t total_hits = 0;
    for (const auto& hits : local_hits) total_hits += hits.size();
    std::vector<uint32_t> hits;
    hits.reserve(total_hits);
    for (auto& thread_hits : local_hits) {
      hits.insert(hits.end(), thread_hits.begin(), thread_hits.end());
      thread_hits.clear();
      thread_hits.shrink_to_fit();
    }
    local_hits.clear();
    std::sort(hits.begin(), hits.end());

    std::vector<VoxelCount> density;
    density.reserve(hits.size());
    for (size_t begin = 0; begin < hits.size();) {
      size_t end = begin + 1;
      while (end < hits.size() && hits[end] == hits[begin]) ++end;
      density.push_back({hits[begin], static_cast<uint32_t>(end - begin)});
      begin = end;
    }
    hits.clear();
    hits.shrink_to_fit();

    const double exposure = percentile_log(density, 0.998);
    const size_t plane = static_cast<size_t>(options.resolution) * options.resolution;
    std::vector<Candidate> candidates;
    candidates.reserve(density.size());
    for (const VoxelCount& voxel_density : density) {
      const float normalized = normalized_density(voxel_density.count, exposure);
      const uint32_t z_index = voxel_density.voxel / plane;
      const uint32_t remainder = voxel_density.voxel % plane;
      const uint32_t y_index = remainder / options.resolution;
      const float y = static_cast<float>(y_index) / std::max(1u, options.resolution - 1);
      const float z = static_cast<float>(z_index) / std::max(1u, options.resolution - 1);
      const float hue_mix = 0.58f * y + 0.42f * z;
      const float luminance = 0.38f + 0.62f * std::sqrt(normalized);
      const float red = std::clamp(luminance * (0.20f + 0.72f * hue_mix), 0.0f, 1.0f);
      const float green = std::clamp(luminance * (0.91f - 0.23f * hue_mix), 0.0f, 1.0f);
      const float blue = std::clamp(luminance * (1.04f - 0.05f * hue_mix), 0.0f, 1.0f);

      // Weighted reservoir. Dense folding dominates, but a small floor keeps
      // wispy low-density branches and internal bridges.
      const double weight = 0.025 + 0.975 * std::pow(normalized, 1.8f);
      const double random = std::max(1e-12, random01(voxel_density.voxel, 8));
      const double key = -std::log(random) / weight;
      candidates.push_back({voxel_density.voxel, red, green, blue, normalized, key});
    }

    if (candidates.size() > options.max_splats) {
      std::nth_element(candidates.begin(), candidates.begin() + options.max_splats,
                       candidates.end(), [](const Candidate& a, const Candidate& b) {
                         return a.selection_key < b.selection_key;
                       });
      candidates.resize(options.max_splats);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return a.voxel < b.voxel;
    });

    write_ply(options, candidates);
    std::filesystem::create_directories(options.stats.parent_path());
    std::ofstream stats(options.stats);
    stats << "{\n"
          << "  \"generator\": \"offline-complex-henon-3dgs\",\n"
          << "  \"candidateSamples\": " << options.samples << ",\n"
          << "  \"escapedSamples\": " << escaped.load() << ",\n"
          << "  \"maxIterations\": " << options.iterations << ",\n"
          << "  \"mapPower\": 2,\n"
          << "  \"couplingMagnitude\": " << COUPLING_MAGNITUDE << ",\n"
          << "  \"couplingPhase\": " << COUPLING_PHASE << ",\n"
          << "  \"resolution\": [" << options.resolution << ", "
          << options.resolution << ", " << options.resolution << "],\n"
          << "  \"volumeAxis\": \"oblique-projection-of-c2\",\n"
          << "  \"gaussians\": " << candidates.size() << ",\n"
          << "  \"splatSigma\": " << std::setprecision(8)
          << (FIELD_MAX - FIELD_MIN) / options.resolution * 0.24 << "\n"
          << "}\n";

    std::cerr << "qualified escaping paths " << escaped.load() << "; "
              << density.size() << " occupied XYZ voxels; wrote " << candidates.size()
              << " Hénon gaussians to " << options.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

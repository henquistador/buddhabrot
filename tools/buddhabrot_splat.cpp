// Offline 3D Buddhabrot (Buddhabulb) -> standard 3DGS PLY.
//
// This is not a stack of 2D Mandelbrots. Each parameter c and every orbit z
// are real XYZ vectors. Iteration uses the power-8 spherical Mandelbulb map,
// z <- bulb_power(z, 8) + c. Only escaping 3D paths enter the density volume.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double FIELD_MIN = -1.55;
constexpr double FIELD_MAX = 1.55;
constexpr double PARAM_MIN = -1.22;
constexpr double PARAM_MAX = 1.22;
constexpr uint32_t BULB_POWER = 8;
constexpr float SH_C0 = 0.28209479177387814f;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Options {
  uint64_t samples = 12'000'000;
  uint32_t iterations = 96;
  uint32_t resolution = 864;
  uint32_t min_escape = 5;
  uint32_t max_splats = 1'000'000;
  uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
  std::filesystem::path output = "outputs/buddhabrot/splat.ply";
  std::filesystem::path stats = "public/buddhabrot.json";
};

struct Candidate {
  uint32_t voxel;
  float red;
  float green;
  float blue;
  float brightness;
};

struct VoxelCount {
  uint32_t voxel;
  uint32_t count;
};

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double random01(uint64_t sample, uint64_t lane) {
  const uint64_t bits = splitmix64(sample * 3 + lane + 0x627564646861ULL);
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

double length_squared(const Vec3& v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

Vec3 bulb_power(const Vec3& value) {
  const double radius2 = length_squared(value);
  if (radius2 < 1e-30) return {};

  const double radius = std::sqrt(radius2);
  const double theta = std::acos(std::clamp(value.z / radius, -1.0, 1.0));
  const double phi = std::atan2(value.y, value.x);
  const double radius_power = std::pow(radius, BULB_POWER);
  const double powered_theta = theta * BULB_POWER;
  const double powered_phi = phi * BULB_POWER;
  const double sin_theta = std::sin(powered_theta);

  return {
      radius_power * sin_theta * std::cos(powered_phi),
      radius_power * sin_theta * std::sin(powered_phi),
      radius_power * std::cos(powered_theta),
  };
}

Vec3 iterate(const Vec3& value, const Vec3& c) {
  const Vec3 powered = bulb_power(value);
  return {powered.x + c.x, powered.y + c.y, powered.z + c.z};
}

uint32_t escape_time(const Vec3& c, uint32_t max_iterations) {
  Vec3 value;
  Vec3 checkpoint;
  uint32_t checkpoint_span = 8;
  uint32_t since_checkpoint = 0;

  for (uint32_t step = 0; step < max_iterations; ++step) {
    value = iterate(value, c);
    if (length_squared(value) > 4.0) return step + 1;

    // Stop converged attracting cycles without pretending they escaped.
    ++since_checkpoint;
    const Vec3 delta{value.x - checkpoint.x, value.y - checkpoint.y, value.z - checkpoint.z};
    if (since_checkpoint > 4 && length_squared(delta) < 1e-26) return 0;
    if (since_checkpoint >= checkpoint_span) {
      checkpoint = value;
      since_checkpoint = 0;
      checkpoint_span = std::min(checkpoint_span * 2u, 32u);
    }
  }
  return 0;
}

void add_orbit(std::vector<uint32_t>& hits, uint32_t resolution,
               const Vec3& c, uint32_t escape) {
  Vec3 value;
  const double scale = resolution / (FIELD_MAX - FIELD_MIN);
  const size_t plane = static_cast<size_t>(resolution) * resolution;

  // Skip z1=c. It is uniformly sampled parameter space, not fractal structure.
  for (uint32_t step = 0; step < escape; ++step) {
    value = iterate(value, c);
    if (step == 0) continue;
    if (value.x < FIELD_MIN || value.x >= FIELD_MAX ||
        value.y < FIELD_MIN || value.y >= FIELD_MAX ||
        value.z < FIELD_MIN || value.z >= FIELD_MAX) continue;

    const uint32_t x = static_cast<uint32_t>((value.x - FIELD_MIN) * scale);
    const uint32_t y = static_cast<uint32_t>((value.y - FIELD_MIN) * scale);
    const uint32_t z = static_cast<uint32_t>((value.z - FIELD_MIN) * scale);
    if (x >= resolution || y >= resolution || z >= resolution) continue;
    const size_t voxel = static_cast<size_t>(z) * plane +
                         static_cast<size_t>(y) * resolution + x;
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
  if (count == 0) return 0.0f;
  const double value = std::min(1.0, std::log1p(static_cast<double>(count)) / exposure);
  return static_cast<float>(std::pow(value, 0.72));
}

void append_float(std::ofstream& output, float value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_ply(const Options& options, const std::vector<Candidate>& splats) {
  std::filesystem::create_directories(options.output.parent_path());
  std::ofstream output(options.output, std::ios::binary);
  if (!output) throw std::runtime_error("could not open output PLY");

  output << "ply\nformat binary_little_endian 1.0\n"
         << "comment offline power-8 3D Buddhabulb escape-orbit density\n"
         << "element vertex " << splats.size() << "\n";
  const char* fields[] = {
      "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
      "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  for (const char* field : fields) output << "property float " << field << "\n";
  output << "end_header\n";

  const size_t plane = static_cast<size_t>(options.resolution) * options.resolution;
  const float voxel_world = static_cast<float>((FIELD_MAX - FIELD_MIN) / options.resolution);
  const float sigma = voxel_world * 0.19f;
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
    const float alpha = std::clamp(0.16f + 0.46f * std::sqrt(splat.brightness), 0.01f, 0.68f);
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
  if (options.resolution > 1024) throw std::runtime_error("resolution must be <= 1024");
  const uint64_t voxel_count = static_cast<uint64_t>(options.resolution) *
                               options.resolution * options.resolution;
  if (voxel_count > UINT32_MAX) throw std::runtime_error("resolution exceeds 32-bit sparse index");
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

    std::cerr << "sampling " << options.samples << " XYZ parameters with power "
              << BULB_POWER << " at " << options.iterations << " max iterations on "
              << options.threads << " threads into " << options.resolution << "^3 voxels\n";

    for (uint32_t thread = 0; thread < options.threads; ++thread) {
      local_hits[thread].reserve(options.samples / options.threads / 2);
      workers.emplace_back([&, thread] {
        constexpr uint64_t CHUNK = 128;
        while (true) {
          const uint64_t begin = cursor.fetch_add(CHUNK);
          if (begin >= options.samples) break;
          const uint64_t end = std::min(options.samples, begin + CHUNK);
          for (uint64_t sample = begin; sample < end; ++sample) {
            const Vec3 c{
                PARAM_MIN + random01(sample, 0) * (PARAM_MAX - PARAM_MIN),
                PARAM_MIN + random01(sample, 1) * (PARAM_MAX - PARAM_MIN),
                PARAM_MIN + random01(sample, 2) * (PARAM_MAX - PARAM_MIN),
            };
            const uint32_t escape = escape_time(c, options.iterations);
            if (escape >= options.min_escape) {
              ++escaped;
              add_orbit(local_hits[thread], options.resolution, c, escape);
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
    std::vector<Candidate> candidates;
    candidates.reserve(options.max_splats * 2);
    const size_t plane = static_cast<size_t>(options.resolution) * options.resolution;

    for (const VoxelCount& voxel_density : density) {
      const uint32_t voxel = voxel_density.voxel;
      const float normalized = normalized_density(voxel_density.count, exposure);
      if (normalized < 0.055f) continue;

      const uint32_t z_index = voxel / plane;
      const float height = static_cast<float>(z_index) / (options.resolution - 1);
      const float magenta = 1.0f - height;
      const float red = std::clamp(normalized * (0.18f + 0.74f * magenta), 0.0f, 1.0f);
      const float green = std::clamp(normalized * (0.62f + 0.30f * height), 0.0f, 1.0f);
      const float blue = std::clamp(normalized * (1.08f - 0.10f * magenta), 0.0f, 1.0f);
      const float brightness = std::max({red, green, blue});
      candidates.push_back({voxel, red, green, blue, brightness});
    }

    if (candidates.size() > options.max_splats) {
      std::nth_element(candidates.begin(), candidates.begin() + options.max_splats,
                       candidates.end(), [](const Candidate& a, const Candidate& b) {
                         return a.brightness > b.brightness;
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
          << "  \"generator\": \"offline-buddhabulb-3dgs\",\n"
          << "  \"candidateSamples\": " << options.samples << ",\n"
          << "  \"escapedSamples\": " << escaped.load() << ",\n"
          << "  \"maxIterations\": " << options.iterations << ",\n"
          << "  \"mapPower\": " << BULB_POWER << ",\n"
          << "  \"resolution\": [" << options.resolution << ", "
          << options.resolution << ", " << options.resolution << "],\n"
          << "  \"volumeAxis\": \"mandelbulb-x-y-z\",\n"
          << "  \"gaussians\": " << candidates.size() << ",\n"
          << "  \"splatSigma\": " << std::setprecision(8)
          << (FIELD_MAX - FIELD_MIN) / options.resolution * 0.19 << "\n"
          << "}\n";

    std::cerr << "qualified escaping paths " << escaped.load() << "; wrote "
              << candidates.size() << " tiny XYZ gaussians to " << options.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

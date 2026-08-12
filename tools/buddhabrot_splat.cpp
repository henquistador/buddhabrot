// Offline Buddhabrot orbit-time volume -> standard 3DGS PLY.
//
// X/Y are the actual complex orbit position. Z is continuous normalized orbit
// progress from z0 to escape. This preserves the canonical Buddhabrot in front
// projection while rotation reveals the trajectories, rather than copied image
// planes or a solid of revolution.

#include <algorithm>
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

constexpr double X_MIN = -2.2;
constexpr double X_MAX = 1.2;
constexpr double Y_MIN = -1.7;
constexpr double Y_MAX = 1.7;
constexpr float VOLUME_DEPTH = 1.15f;
constexpr float SH_C0 = 0.28209479177387814f;

struct Options {
  uint64_t samples = 12'000'000;
  uint32_t iterations = 1'048'576;
  uint32_t resolution = 1'600;
  uint32_t depth = 256;
  uint32_t min_escape = 16;
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
  const uint64_t bits = splitmix64(sample * 2 + lane + 0x627564646861ULL);
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

bool known_interior(double cr, double ci) {
  const double bulb = (cr + 1.0) * (cr + 1.0) + ci * ci;
  const double dx = cr - 0.25;
  const double q = dx * dx + ci * ci;
  return bulb <= 0.0625 || q * (q + dx) <= 0.25 * ci * ci;
}

uint32_t escape_time(double cr, double ci, uint32_t max_iterations) {
  if (known_interior(cr, ci)) return 0;

  double zr = 0.0;
  double zi = 0.0;
  double checkpoint_r = 0.0;
  double checkpoint_i = 0.0;
  uint32_t checkpoint_span = 32;
  uint32_t since_checkpoint = 0;

  for (uint32_t step = 0; step < max_iterations; ++step) {
    const double next_r = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = next_r;
    if (zr * zr + zi * zi > 4.0) return step + 1;

    ++since_checkpoint;
    const double dr = zr - checkpoint_r;
    const double di = zi - checkpoint_i;
    if (since_checkpoint > 8 && dr * dr + di * di < 1e-28) return 0;
    if (since_checkpoint >= checkpoint_span) {
      checkpoint_r = zr;
      checkpoint_i = zi;
      since_checkpoint = 0;
      checkpoint_span = std::min(checkpoint_span * 2u, 4096u);
    }
  }
  return 0;
}

void add_orbit(std::vector<uint32_t>& hits, const Options& options,
               double cr, double ci, uint32_t escape) {
  const size_t pixels = static_cast<size_t>(options.resolution) * options.resolution;
  double zr = 0.0;
  double zi = 0.0;

  for (uint32_t step = 0; step < escape; ++step) {
    const double next_r = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = next_r;
    if (zr < X_MIN || zr >= X_MAX || zi < Y_MIN || zi >= Y_MAX) continue;

    const uint32_t px = static_cast<uint32_t>((zr - X_MIN) / (X_MAX - X_MIN) * options.resolution);
    const uint32_t py = static_cast<uint32_t>((Y_MAX - zi) / (Y_MAX - Y_MIN) * options.resolution);
    if (px >= options.resolution || py >= options.resolution) continue;
    const uint32_t pz = std::min(options.depth - 1,
        static_cast<uint32_t>(static_cast<uint64_t>(step) * options.depth / escape));
    const uint32_t voxel = static_cast<uint32_t>(static_cast<size_t>(pz) * pixels +
                                                  static_cast<size_t>(py) * options.resolution + px);
    hits.push_back(voxel);

    const uint32_t mirror_y = options.resolution - 1 - py;
    if (mirror_y != py) {
      hits.push_back(static_cast<uint32_t>(static_cast<size_t>(pz) * pixels +
                                          static_cast<size_t>(mirror_y) * options.resolution + px));
    }
  }
}

double percentile_log(const std::vector<uint32_t>& density, double quantile) {
  std::vector<float> values;
  values.reserve(density.size());
  for (const uint32_t count : density) {
    if (count != 0) values.push_back(std::log1p(static_cast<float>(count)));
  }
  if (values.empty()) return 1.0;
  const size_t index = std::min(values.size() - 1,
      static_cast<size_t>(quantile * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return std::max(1e-6f, values[index]);
}

float normalized_density(uint32_t count, double exposure) {
  const double value = std::min(1.0, std::log1p(static_cast<double>(count)) / exposure);
  return static_cast<float>(std::pow(value, 3.0));
}

void append_float(std::ofstream& output, float value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_ply(const Options& options, const std::vector<Candidate>& splats) {
  std::filesystem::create_directories(options.output.parent_path());
  std::ofstream output(options.output, std::ios::binary);
  if (!output) throw std::runtime_error("could not open output PLY");

  output << "ply\nformat binary_little_endian 1.0\n"
         << "comment offline Buddhabrot XY orbit and continuous time depth\n"
         << "element vertex " << splats.size() << "\n";
  const char* fields[] = {
      "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
      "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  for (const char* field : fields) output << "property float " << field << "\n";
  output << "end_header\n";

  const size_t pixels = static_cast<size_t>(options.resolution) * options.resolution;
  const float sigma = static_cast<float>((X_MAX - X_MIN) / options.resolution * 0.65);
  const float log_sigma = std::log(sigma);

  for (const Candidate& splat : splats) {
    const uint32_t pz = splat.voxel / pixels;
    const uint32_t pixel = splat.voxel % pixels;
    const uint32_t py = pixel / options.resolution;
    const uint32_t px = pixel % options.resolution;
    const float x = static_cast<float>(X_MIN + (px + 0.5) / options.resolution * (X_MAX - X_MIN));
    const float y = static_cast<float>(Y_MAX - (py + 0.5) / options.resolution * (Y_MAX - Y_MIN));
    const float z = (static_cast<float>(pz) / std::max(1u, options.depth - 1) - 0.5f) * VOLUME_DEPTH;
    const float alpha = std::clamp(
        0.015f + 0.65f * std::pow(splat.density, 1.25f), 0.01f, 0.665f);
    const float opacity = std::log(alpha / (1.0f - alpha));
    const float values[] = {
        x, y, z, 0.0f, 0.0f, 0.0f,
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
    else if (arg == "--depth") options.depth = std::stoul(next());
    else if (arg == "--min-escape") options.min_escape = std::stoul(next());
    else if (arg == "--max-splats") options.max_splats = std::stoul(next());
    else if (arg == "--threads") options.threads = std::max(1ul, std::stoul(next()));
    else if (arg == "--output") options.output = next();
    else if (arg == "--stats") options.stats = next();
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (options.resolution == 0 || options.depth == 0) {
    throw std::runtime_error("resolution and depth must be positive");
  }
  const uint64_t voxels = static_cast<uint64_t>(options.resolution) * options.resolution * options.depth;
  if (voxels > UINT32_MAX) throw std::runtime_error("volume exceeds 32-bit sparse index");
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

    std::cerr << "sampling " << options.samples << " complex parameters at "
              << options.iterations << " max iterations on " << options.threads
              << " threads into " << options.resolution << "^2 x " << options.depth << " voxels\n";

    for (uint32_t thread = 0; thread < options.threads; ++thread) {
      local_hits[thread].reserve(options.samples / options.threads);
      workers.emplace_back([&, thread] {
        constexpr uint64_t CHUNK = 128;
        while (true) {
          const uint64_t begin = cursor.fetch_add(CHUNK);
          if (begin >= options.samples) break;
          const uint64_t end = std::min(options.samples, begin + CHUNK);
          for (uint64_t sample = begin; sample < end; ++sample) {
            const double cr = X_MIN + random01(sample, 0) * (X_MAX - X_MIN);
            const double ci = random01(sample, 1) * Y_MAX;
            const uint32_t escape = escape_time(cr, ci, options.iterations);
            if (escape >= options.min_escape) {
              ++escaped;
              add_orbit(local_hits[thread], options, cr, ci, escape);
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

    const size_t pixels = static_cast<size_t>(options.resolution) * options.resolution;
    std::vector<uint32_t> projected_density(pixels, 0);
    std::vector<uint32_t> chosen_voxel(pixels, UINT32_MAX);
    std::vector<double> chosen_depth_key(pixels, INFINITY);
    for (const VoxelCount& voxel_density : density) {
      const uint32_t pixel = voxel_density.voxel % pixels;
      uint32_t& projected = projected_density[pixel];
      projected = UINT32_MAX - projected < voxel_density.count
          ? UINT32_MAX
          : projected + voxel_density.count;
      const double random = std::max(1e-12, random01(voxel_density.voxel, 6));
      const double key = -std::log(random) / voxel_density.count;
      if (key < chosen_depth_key[pixel]) {
        chosen_depth_key[pixel] = key;
        chosen_voxel[pixel] = voxel_density.voxel;
      }
    }
    const double exposure = percentile_log(projected_density, 0.998);
    std::vector<Candidate> candidates;
    candidates.reserve(pixels);
    const size_t occupied_pixels = std::count_if(
        projected_density.begin(), projected_density.end(), [](uint32_t count) { return count != 0; });
    const double interior_probability = occupied_pixels == 0
        ? 0.0
        : std::min(1.0, options.max_splats * 0.15 / static_cast<double>(occupied_pixels));

    for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
      if (chosen_voxel[pixel] == UINT32_MAX) continue;
      // Rank and shade using the canonical front-projected Buddhabrot density.
      // The voxel still retains its own continuous orbit-time depth.
      const float normalized = normalized_density(projected_density[pixel], exposure);
      const uint32_t pz = chosen_voxel[pixel] / pixels;
      const float time = static_cast<float>(pz) / std::max(1u, options.depth - 1);
      const float luminance = 0.42f + 0.58f * std::sqrt(normalized);
      const float red = std::clamp(luminance * (0.16f + 0.74f * time), 0.0f, 1.0f);
      const float green = std::clamp(luminance * (0.94f - 0.30f * time), 0.0f, 1.0f);
      const float blue = std::clamp(luminance * (1.02f - 0.04f * time), 0.0f, 1.0f);
      // Reserve about 150K occupied pixels for faint inner trajectories. Fill
      // the rest strictly by projected density for a crisp Buddha silhouette.
      const double random = random01(pixel, 7);
      const bool reserve_interior = random01(pixel, 8) < interior_probability;
      const double key = reserve_interior
          ? -2.0 - random
          : -static_cast<double>(normalized) - random * 1e-7;
      candidates.push_back({chosen_voxel[pixel], red, green, blue, normalized, key});
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
          << "  \"generator\": \"offline-buddhabrot-orbit-time-3dgs\",\n"
          << "  \"candidateSamples\": " << options.samples << ",\n"
          << "  \"mirroredSamples\": " << options.samples * 2 << ",\n"
          << "  \"escapedSamples\": " << escaped.load() << ",\n"
          << "  \"maxIterations\": " << options.iterations << ",\n"
          << "  \"mapPower\": 2,\n"
          << "  \"resolution\": [" << options.resolution << ", "
          << options.resolution << ", " << options.depth << "],\n"
          << "  \"volumeAxis\": \"normalized-orbit-progress\",\n"
          << "  \"volumeDepth\": " << VOLUME_DEPTH << ",\n"
          << "  \"gaussians\": " << candidates.size() << ",\n"
          << "  \"splatSigma\": " << std::setprecision(8)
          << (X_MAX - X_MIN) / options.resolution * 0.65 << "\n"
          << "}\n";

    std::cerr << "qualified escaping paths " << escaped.load() << "; "
              << density.size() << " occupied voxels; wrote " << candidates.size()
              << " tiny orbit-time gaussians to " << options.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

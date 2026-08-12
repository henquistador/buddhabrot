// Offline Buddhabrot -> standard 3D Gaussian Splatting PLY.
//
// Samples the upper half of the c-plane, mirrors every escaped orbit, and
// accumulates an XYT density volume, where T is normalized orbit progress from
// z0 to escape. The maximum escape test is deliberately
// independent from output resolution: --iterations 1048576 is a real one-
// million-step cap, not a label or a progressive browser approximation.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double X_MIN = -2.2;
constexpr double X_MAX = 1.2;
constexpr double Y_MIN = -1.7;
constexpr double Y_MAX = 1.7;
constexpr float SH_C0 = 0.28209479177387814f;

struct Options {
  uint64_t samples = 10'000'000;
  uint32_t iterations = 1'048'576;
  uint32_t width = 1200;
  uint32_t height = 1200;
  uint32_t depth = 48;
  uint32_t max_splats = 600'000;
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

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double random01(uint64_t index, uint64_t lane) {
  const uint64_t bits = splitmix64(index * 2 + lane + 0x627564646861ULL);
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
  uint32_t since_checkpoint = 0;
  uint32_t checkpoint_span = 32;

  for (uint32_t step = 0; step < max_iterations; ++step) {
    const double next_r = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = next_r;
    if (zr * zr + zi * zi > 4.0) return step + 1;

    // Attracting cycles converge numerically. This prevents interior bulbs
    // outside the analytic cardioid tests from consuming the full 1M cap.
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

void add_orbit(std::atomic<uint32_t>* histogram, uint32_t width,
               uint32_t height, uint32_t depth, double cr, double ci,
               uint32_t escape) {
  if (escape < 16) return;
  const size_t pixels = static_cast<size_t>(width) * height;
  double zr = 0.0;
  double zi = 0.0;

  for (uint32_t step = 0; step < escape; ++step) {
    const double next_r = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = next_r;
    if (zr < X_MIN || zr >= X_MAX || zi < Y_MIN || zi >= Y_MAX) continue;

    const uint32_t px = static_cast<uint32_t>((zr - X_MIN) / (X_MAX - X_MIN) * width);
    const uint32_t py = static_cast<uint32_t>((Y_MAX - zi) / (Y_MAX - Y_MIN) * height);
    if (px >= width || py >= height) continue;
    const size_t pixel = static_cast<size_t>(py) * width + px;
    const uint32_t pz = std::min(depth - 1,
        static_cast<uint32_t>(static_cast<uint64_t>(step) * depth / escape));
    ++histogram[static_cast<size_t>(pz) * pixels + pixel];

    // Samples only cover Im(c) >= 0. Mirror the conjugate path exactly.
    const uint32_t mirror_y = height - 1 - py;
    const size_t mirror = static_cast<size_t>(mirror_y) * width + px;
    if (mirror != pixel) {
      ++histogram[static_cast<size_t>(pz) * pixels + mirror];
    }
  }
}

double percentile_log(const std::atomic<uint32_t>* histogram, size_t voxels,
                      double quantile) {
  std::vector<float> values;
  values.reserve(voxels / 8);
  for (size_t i = 0; i < voxels; ++i) {
    const uint32_t count = histogram[i].load(std::memory_order_relaxed);
    if (count != 0) {
      values.push_back(std::log1p(static_cast<float>(count)));
    }
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
  return static_cast<float>(std::pow(value, 0.78));
}

void append_float(std::ofstream& output, float value) {
  static_assert(sizeof(float) == 4);
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_ply(const Options& options, const std::vector<Candidate>& splats) {
  std::filesystem::create_directories(options.output.parent_path());
  std::ofstream output(options.output, std::ios::binary);
  if (!output) throw std::runtime_error("could not open output PLY");

  output << "ply\nformat binary_little_endian 1.0\n"
         << "comment offline Buddhabrot, max_iterations " << options.iterations << "\n"
         << "element vertex " << splats.size() << "\n";
  const char* fields[] = {
      "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
      "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  for (const char* field : fields) output << "property float " << field << "\n";
  output << "end_header\n";

  const float pixel_world = static_cast<float>((X_MAX - X_MIN) / options.width);
  const float sigma = pixel_world * 0.24f;
  const float log_sigma = std::log(sigma);
  constexpr float DEPTH_EXTENT = 1.45f;

  for (const Candidate& splat : splats) {
    const uint32_t pixel = splat.voxel % (options.width * options.height);
    const uint32_t pz = splat.voxel / (options.width * options.height);
    const uint32_t px = pixel % options.width;
    const uint32_t py = pixel / options.width;
    const float x = static_cast<float>(X_MIN + (px + 0.5) / options.width * (X_MAX - X_MIN));
    const float y = static_cast<float>(Y_MAX - (py + 0.5) / options.height * (Y_MAX - Y_MIN));
    const float z = (static_cast<float>(pz) / std::max(1u, options.depth - 1) - 0.5f) * DEPTH_EXTENT;
    const float alpha = std::clamp(0.50f + 0.47f * std::sqrt(splat.brightness), 0.01f, 0.97f);
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
    else if (arg == "--resolution") options.width = options.height = std::stoul(next());
    else if (arg == "--depth") options.depth = std::stoul(next());
    else if (arg == "--max-splats") options.max_splats = std::stoul(next());
    else if (arg == "--threads") options.threads = std::max(1ul, std::stoul(next()));
    else if (arg == "--output") options.output = next();
    else if (arg == "--stats") options.stats = next();
    else throw std::runtime_error("unknown argument: " + arg);
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const size_t pixels = static_cast<size_t>(options.width) * options.height;
    const size_t voxels = pixels * options.depth;
    std::atomic<uint64_t> cursor{0};
    std::atomic<uint64_t> escaped{0};
    auto histogram = std::make_unique<std::atomic<uint32_t>[]>(voxels);
    for (size_t i = 0; i < voxels; ++i) histogram[i].store(0, std::memory_order_relaxed);
    std::vector<std::thread> workers;
    workers.reserve(options.threads);

    std::cerr << "sampling " << options.samples << " c values at "
              << options.iterations << " max iterations on " << options.threads
              << " threads into " << options.width << "x" << options.height
              << "x" << options.depth << " voxels\n";

    for (uint32_t thread = 0; thread < options.threads; ++thread) {
      workers.emplace_back([&] {
        constexpr uint64_t CHUNK = 128;
        while (true) {
          const uint64_t begin = cursor.fetch_add(CHUNK);
          if (begin >= options.samples) break;
          const uint64_t end = std::min(options.samples, begin + CHUNK);
          for (uint64_t sample = begin; sample < end; ++sample) {
            const double cr = X_MIN + random01(sample, 0) * (X_MAX - X_MIN);
            const double ci = random01(sample, 1) * Y_MAX;
            const uint32_t escape = escape_time(cr, ci, options.iterations);
            if (escape >= 16) {
              ++escaped;
              add_orbit(histogram.get(), options.width, options.height,
                        options.depth, cr, ci, escape);
            }
          }
        }
      });
    }
    for (auto& worker : workers) worker.join();

    const double exposure = percentile_log(histogram.get(), voxels, 0.9985);
    std::vector<Candidate> candidates;
    candidates.reserve(options.max_splats * 2);

    for (uint32_t voxel = 0; voxel < voxels; ++voxel) {
      const uint32_t count = histogram[voxel].load(std::memory_order_relaxed);
      const float density = normalized_density(count, exposure);
      const uint32_t pz = voxel / pixels;
      const float t = static_cast<float>(pz) / std::max(1u, options.depth - 1);
      const float red = std::clamp(density * (0.26f + 0.92f * t), 0.0f, 1.0f);
      const float green = std::clamp(density * (0.92f - 0.28f * t), 0.0f, 1.0f);
      const float blue = std::clamp(density * (1.10f - 0.12f * t), 0.0f, 1.0f);
      const float brightness = std::max({red, green, blue});
      if (brightness >= 0.07f) candidates.push_back({voxel, red, green, blue, brightness});
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
          << "  \"generator\": \"offline-buddhabrot-3dgs\",\n"
          << "  \"candidateSamples\": " << options.samples << ",\n"
          << "  \"mirroredSamples\": " << options.samples * 2 << ",\n"
          << "  \"escapedSamples\": " << escaped.load() << ",\n"
          << "  \"maxIterations\": " << options.iterations << ",\n"
          << "  \"resolution\": [" << options.width << ", " << options.height
          << ", " << options.depth << "],\n"
          << "  \"volumeAxis\": \"normalized-orbit-progress\",\n"
          << "  \"volumeDepth\": 1.45,\n"
          << "  \"gaussians\": " << candidates.size() << ",\n"
          << "  \"splatSigma\": " << std::setprecision(8)
          << (X_MAX - X_MIN) / options.width * 0.24 << "\n"
          << "}\n";

    std::cerr << "escaped " << escaped.load() << " samples; wrote "
              << candidates.size() << " tiny gaussians to " << options.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

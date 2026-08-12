# Offline Buddhabrot Splat

A precomputed Buddhabrot density exposure presented as a standard 3D Gaussian
Splatting asset. The production artifact tests ten million candidate `c` values
against a real 1,048,576-iteration escape cap, mirrors the paths for twenty million
effective samples, then emits 600,000 tiny isotropic Gaussian dots into an XYT
volume. X/Y store orbit position; depth stores normalized progress from origin
to escape. A head-on projection remains a Buddhabrot, while rotation exposes
the trajectories inside it.

Fractal generation is entirely offline. The browser only downloads and presents
the 1.7 MB `public/buddhabrot.spz` artifact with Spark. The earlier interactive
orbit experiment remains available at `/orbit`.

## Regenerate the artifact

Requires `clang++`, Node.js, and npm. Defaults produce the checked-in asset:

```bash
npm run generate:splat
```

Override the workload with environment variables:

```bash
BUDDHABROT_SAMPLES=5000000 \
BUDDHABROT_ITERATIONS=1048576 \
BUDDHABROT_RESOLUTION=1000 \
BUDDHABROT_DEPTH=64 \
BUDDHABROT_MAX_SPLATS=800000 \
npm run generate:splat
```

The generator writes the uncompressed standard 3DGS PLY to
`outputs/buddhabrot/splat.ply`, then uses `@playcanvas/splat-transform` to write
the web-delivery SPZ to `public/buddhabrot.spz`.

## Run locally

Requires Node.js 22.13+ and a WebGPU-capable browser.

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

For testing from another device on the same LAN, expose the development server:

```bash
npm run dev -- --host 0.0.0.0
```

Then open `http://<your-computer-lan-ip>:3000`. Your firewall must allow the
connection. The hosted production URL avoids LAN and firewall setup.

## Viewer controls

- Drag: rotate the 3D volume
- Shift-drag or right-drag: pan
- Scroll: zoom
- Double-click: reset the camera

## Verify

```bash
npm test
```

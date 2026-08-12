# Offline 3D Buddhabrot Splat

A precomputed, genuinely three-dimensional Buddhabrot presented as a standard
3D Gaussian Splatting asset. The generator samples 12 million independent XYZ
parameters and iterates a quadratic quaternion slice. Its complex cross-section
is the familiar Mandelbrot map, while continuously angled escape paths fill real
XYZ space. The paths enter a sparse 864³ density lattice, from which a weighted
spatial reservoir preserves one million tiny, translucent dots across both dense
and faint interior regions. No planes are copied or extruded.

Fractal generation is entirely offline. The browser only downloads and presents
the 3.6 MB `public/buddhabrot.spz` artifact with Spark. The earlier interactive
orbit experiment remains available at `/orbit`.

## Regenerate the artifact

Requires `clang++`, Node.js, and npm. Defaults produce the checked-in asset:

```bash
npm run generate:splat
```

Override the workload with environment variables:

```bash
BUDDHABROT_SAMPLES=20000000 \
BUDDHABROT_ITERATIONS=128 \
BUDDHABROT_RESOLUTION=256 \
BUDDHABROT_MIN_ESCAPE=5 \
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

- Drag: rotate the real XYZ volume
- Shift-drag or right-drag: pan
- Scroll: zoom
- Double-click: reset the camera

## Verify

```bash
npm test
```

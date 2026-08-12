# Offline Complex Hénon Escape Splat

A precomputed Buddhabrot-like escape cloud presented as a standard 3D Gaussian
Splatting asset. The generator iterates the complex Hénon map
`(z,w) → (z² + c - a·w, z)` for 12 million candidate values of `c`, using the
fixed complex coupling `a = 0.22 exp(0.65i)`. Each path lives in four real
dimensions. A fixed oblique projection maps the coupled C² state into a sparse
864³ XYZ lattice, then emits one million tiny translucent dots. No coordinate is
orbit time, and no image planes are copied, stacked, or revolved.

Fractal generation is entirely offline. The browser only downloads and presents
the 4.1 MB `public/henon-buddhabrot.spz` artifact with Spark. The earlier interactive
orbit experiment remains available at `/orbit`.

## Regenerate the artifact

Requires `clang++`, Node.js, and npm. Defaults produce the checked-in asset:

```bash
npm run generate:splat
```

Override the workload with environment variables:

```bash
BUDDHABROT_SAMPLES=20000000 \
BUDDHABROT_ITERATIONS=512 \
BUDDHABROT_RESOLUTION=864 \
BUDDHABROT_MIN_ESCAPE=8 \
BUDDHABROT_MAX_SPLATS=800000 \
npm run generate:splat
```

The generator writes the uncompressed standard 3DGS PLY to
`outputs/buddhabrot/splat.ply`, then uses `@playcanvas/splat-transform` to write
the web-delivery SPZ to `public/henon-buddhabrot.spz`.

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

# Orbit Lab

A WebGPU Mandelbrot iterator. Move the pointer through the complex plane and
watch nearby `z = z² + c` orbits accumulate as points. No line segments are
drawn.

The renderer keeps orbit generation, point drawing, trail accumulation, and
tone mapping on the GPU. Automatic load control adjusts the seed count to hold
60 FPS. Orbit depth ranges from 32 to 8,192 iterations.

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

## Controls

- Move pointer: choose `c`
- Scroll: zoom
- Shift + drag: pan
- Orbit depth: 32–8,192 iterations, with a fixed GPU point budget
- Seed field: maximum nearby trajectories
- Trail memory: accumulation fade
- Point size: 0.25–4 px; larger values use instanced circular quads
- Size slope: −3 to +3; positive values grow later iteration points
- Halo: enable or disable neighboring-pixel bloom
- Auto-load: adapt seed count to the 60 FPS target

## Verify

```bash
npm test
```

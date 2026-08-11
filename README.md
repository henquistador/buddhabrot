# Buddhabrot Splat Lab

A live WebGPU Gaussian-splat Buddhabrot. Candidate `c` values are iterated in
parallel. Bounded paths are rejected; each step of every escaping orbit becomes
an anisotropic Gaussian. Orbit time becomes the third axis, producing a 3D
volume instead of a flat exposure.

Orbit rejection, splat generation, projection, exposure accumulation, bloom,
and tone mapping stay on the GPU. Automatic load control adjusts the candidate
seed count toward 60 FPS. No precomputed model or texture is loaded.

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

- Drag: orbit the 3D volume
- Scroll: dolly the camera
- Orbit depth: 32–2,048 iterations, under a fixed GPU splat budget
- Candidate seeds: maximum `c` values tested per frame
- Volume depth: scale orbit time on the third axis; zero gives classic 2D
- Seed radius: size of the sampled complex-plane square
- Splat scale and tangent stretch: Gaussian covariance controls
- Exposure memory: accumulation fade
- Bloom and auto orbit: display controls
- Auto-load: adapt seed count to the 60 FPS target

## Verify

```bash
npm test
```

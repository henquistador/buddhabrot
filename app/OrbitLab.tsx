"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Stats = {
  fps: number;
  samples: number;
  throughput: number;
  particles: number;
};

type EngineSettings = {
  iterations: number;
  persistence: number;
};

const MAX_ITERATIONS = 50_000_000;
const CURSOR_SEEDS = 100;
const MAX_STREAMS = 1_600;
const CURSOR_DISK_RADIUS = 0.04;
const FRAME_POINT_BUDGET = 100_000;
const MAX_FRAME_POINTS = FRAME_POINT_BUDGET;
const WORKGROUP_SIZE = 64;
const DEFAULT_ZOOM = 3;

const computeShader = /* wgsl */ `
struct Params {
  c: vec2f,
  aspect: f32,
  zoom: f32,
  particleCount: u32,
  iterations: u32,
  batchIterations: u32,
  generation: u32,
  spread: f32,
  pad0: f32,
  center: vec2f,
  pad: vec2f,
}

struct OrbitPoint {
  position: vec2f,
  stepT: f32,
  pad: f32,
}

struct OrbitState {
  z: vec2f,
  sampleC: vec2f,
  step: u32,
  generation: u32,
  cycle: u32,
  alive: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> vertices: array<OrbitPoint>;
@group(0) @binding(2) var<storage, read_write> states: array<OrbitState>;

fn toClip(z: vec2f) -> vec2f {
  let view = vec2f(2.35 * params.aspect, 2.35) / params.zoom;
  return (z - params.center) / view;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let particle = id.x;
  if (particle >= params.particleCount) { return; }

  var state = states[particle];
  for (var localStep = 0u; localStep < params.batchIterations; localStep++) {
    let slot = particle * params.batchIterations + localStep;
    if (state.alive == 0u || state.step >= params.iterations) {
      vertices[slot] = OrbitPoint(vec2f(4.0, 4.0), 0.0, 0.0);
      continue;
    }

    let z = vec2f(
      state.z.x * state.z.x - state.z.y * state.z.y,
      2.0 * state.z.x * state.z.y,
    ) + state.sampleC;
    state.z = z;
    let stepT = f32(state.step) / f32(max(params.iterations - 1u, 1u));
    vertices[slot] = OrbitPoint(toClip(z), stepT, 0.0);
    state.step += 1u;
    if (dot(z, z) > 4.0) { state.alive = 0u; }
  }
  states[particle] = state;
}
`;

const orbitShader = /* wgsl */ `
struct OrbitUniforms {
  alpha: f32,
  hue: f32,
  pad: vec2f,
}
@group(0) @binding(0) var<uniform> style: OrbitUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
}

@vertex
fn vs(
  @location(0) position: vec2f,
  @location(1) stepT: f32,
) -> VSOut {
  var out: VSOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.color = mix(vec3f(0.20, 1.0, 0.60), vec3f(0.30, 0.48, 1.0), stepT + style.hue);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.color * style.alpha, style.alpha);
}
`;

const fullscreenShader = /* wgsl */ `
struct FadeUniforms { fade: f32, gain: f32, pad: vec2f }
@group(0) @binding(0) var previous: texture_2d<f32>;
@group(0) @binding(1) var previousSampler: sampler;
@group(0) @binding(2) var<uniform> settings: FadeUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  let positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VSOut;
  out.position = vec4f(positions[i], 0.0, 1.0);
  out.uv = positions[i] * vec2f(0.5, -0.5) + 0.5;
  return out;
}

@fragment
fn fadeFs(in: VSOut) -> @location(0) vec4f {
  return textureSample(previous, previousSampler, in.uv) * settings.fade;
}

@fragment
fn displayFs(in: VSOut) -> @location(0) vec4f {
  let center = textureSample(previous, previousSampler, in.uv).rgb;
  let raw = center * settings.gain;
  let mapped = vec3f(1.0) - exp(-raw);
  let base = vec3f(0.016, 0.022, 0.018);
  return vec4f(base + mapped, 1.0);
}
`;

function formatCount(value: number) {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

function escapeLabel(cr: number, ci: number, max: number) {
  let zr = 0;
  let zi = 0;
  const probeDepth = Math.min(max, 2048);
  for (let i = 0; i < probeDepth; i++) {
    const nextR = zr * zr - zi * zi + cr;
    zi = 2 * zr * zi + ci;
    zr = nextR;
    if (zr * zr + zi * zi > 4) return `Escapes after ${i + 1} iterations`;
  }
  return max > probeDepth
    ? `No escape in the first ${probeDepth.toLocaleString()} iterations`
    : `Bounded through ${max.toLocaleString()} iterations`;
}

export default function OrbitLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<{ update: (next: Partial<EngineSettings>) => void; reset: () => void } | null>(null);
  const pointRef = useRef({ x: -0.74364, y: 0.13183 });
  const settingsRef = useRef<EngineSettings>({
    iterations: MAX_ITERATIONS,
    persistence: 97,
  });
  const [point, setPoint] = useState(pointRef.current);
  const [settings, setSettings] = useState(settingsRef.current);
  const [stats, setStats] = useState<Stats>({ fps: 0, samples: 0, throughput: 0, particles: 0 });
  const [error, setError] = useState<string | null>(null);

  const patchSettings = useCallback((next: Partial<EngineSettings>) => {
    settingsRef.current = { ...settingsRef.current, ...next };
    setSettings(settingsRef.current);
    engineRef.current?.update(next);
  }, []);

  useEffect(() => {
    let disposed = false;
    let frame = 0;
    let ro: ResizeObserver | null = null;

    async function start() {
      const canvas = canvasRef.current;
      const gpu = (navigator as Navigator & { gpu?: any }).gpu;
      if (!canvas || !gpu) {
        setError("WebGPU is not available here. Open this in current Chrome, Edge, or Safari 26+ with hardware acceleration enabled.");
        return;
      }

      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) {
        setError("No compatible GPU adapter was found. Check that browser hardware acceleration is enabled.");
        return;
      }

      const device = await adapter.requestDevice();
      if (disposed) return;
      const context = canvas.getContext("webgpu") as any;
      const canvasFormat = gpu.getPreferredCanvasFormat();
      context.configure({ device, format: canvasFormat, alphaMode: "opaque" });

      const usage = (globalThis as any).GPUBufferUsage;
      const textureUsage = (globalThis as any).GPUTextureUsage;
      const vertexBuffer = device.createBuffer({
        size: MAX_FRAME_POINTS * 16,
        usage: usage.STORAGE | usage.VERTEX,
      });
      const stateBuffer = device.createBuffer({
        size: MAX_STREAMS * 32,
        usage: usage.STORAGE | usage.COPY_DST,
      });
      const paramsBuffer = device.createBuffer({ size: 64, usage: usage.UNIFORM | usage.COPY_DST });
      const orbitStyleBuffer = device.createBuffer({ size: 16, usage: usage.UNIFORM | usage.COPY_DST });
      const fadeBuffer = device.createBuffer({ size: 16, usage: usage.UNIFORM | usage.COPY_DST });
      const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

      const computeModule = device.createShaderModule({ code: computeShader });
      const orbitModule = device.createShaderModule({ code: orbitShader });
      const fullscreenModule = device.createShaderModule({ code: fullscreenShader });

      const computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: computeModule, entryPoint: "main" },
      });
      const orbitPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: orbitModule,
          entryPoint: "vs",
          buffers: [{
            arrayStride: 16,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x2" },
              { shaderLocation: 1, offset: 8, format: "float32" },
            ],
          }],
        },
        fragment: {
          module: orbitModule,
          entryPoint: "fs",
          targets: [{
            format: "rgba16float",
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          }],
        },
        primitive: { topology: "point-list" },
      });
      const fadePipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: fullscreenModule, entryPoint: "vs" },
        fragment: { module: fullscreenModule, entryPoint: "fadeFs", targets: [{ format: "rgba16float" }] },
        primitive: { topology: "triangle-list" },
      });
      const displayPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: fullscreenModule, entryPoint: "vs" },
        fragment: { module: fullscreenModule, entryPoint: "displayFs", targets: [{ format: canvasFormat }] },
        primitive: { topology: "triangle-list" },
      });

      const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: vertexBuffer } },
          { binding: 2, resource: { buffer: stateBuffer } },
        ],
      });
      const orbitBindGroup = device.createBindGroup({
        layout: orbitPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: orbitStyleBuffer } }],
      });
      let textures: any[] = [];
      let fadeBindGroups: any[] = [];
      let displayBindGroups: any[] = [];
      let textureSize = { width: 0, height: 0 };
      let textureIndex = 0;
      let currentParticles = 0;
      let nextStream = 0;
      let lastSpawnTime = 0;
      let viewCenter = { x: -0.62, y: 0 };
      let zoom = DEFAULT_ZOOM;
      let dragging = false;
      let lastPointer = { x: 0, y: 0 };
      let lastStatTime = performance.now();
      let lastFrameTime = performance.now();
      let smoothFrameMs = 16.7;
      let pageVisible = !document.hidden;

      function spawnCloud(c: { x: number; y: number }) {
        const states = new Float32Array(CURSOR_SEEDS * 8);
        const uintStates = new Uint32Array(states.buffer);
        const maxRadius = CURSOR_DISK_RADIUS / zoom;
        for (let seed = 0; seed < CURSOR_SEEDS; seed++) {
          const angle = Math.random() * Math.PI * 2;
          const radius = Math.sqrt(Math.random()) * maxRadius;
          const offset = seed * 8;
          states[offset + 2] = c.x + Math.cos(angle) * radius;
          states[offset + 3] = c.y + Math.sin(angle) * radius;
          uintStates[offset + 7] = 1;
        }
        device.queue.writeBuffer(stateBuffer, nextStream * 32, states);
        nextStream = (nextStream + CURSOR_SEEDS) % MAX_STREAMS;
        currentParticles = Math.min(MAX_STREAMS, currentParticles + CURSOR_SEEDS);
      }

      function resetStreams() {
        device.queue.writeBuffer(stateBuffer, 0, new Uint8Array(MAX_STREAMS * 32));
        currentParticles = 0;
        nextStream = 0;
        spawnCloud(pointRef.current);
      }

      function makeTextureBindGroup(pipeline: any, texture: any) {
        return device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: fadeBuffer } },
          ],
        });
      }

      function resize() {
        if (!canvas) return;
        const dpr = 1;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width * dpr));
        const height = Math.max(1, Math.round(rect.height * dpr));
        if (width === textureSize.width && height === textureSize.height) return;
        canvas.width = width;
        canvas.height = height;
        textures.forEach((texture) => texture.destroy());
        textures = [0, 1].map(() => device.createTexture({
          size: [width, height],
          format: "rgba16float",
          usage: textureUsage.RENDER_ATTACHMENT | textureUsage.TEXTURE_BINDING,
        }));
        fadeBindGroups = textures.map((texture) => makeTextureBindGroup(fadePipeline, texture));
        displayBindGroups = textures.map((texture) => makeTextureBindGroup(displayPipeline, texture));
        textureSize = { width, height };
        textureIndex = 0;
      }

      function clearAccumulation() {
        if (!textures.length) return;
        const encoder = device.createCommandEncoder();
        for (const texture of textures) {
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: texture.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
          });
          pass.end();
        }
        device.queue.submit([encoder.finish()]);
      }

      const toComplex = (clientX: number, clientY: number) => {
        const rect = canvas.getBoundingClientRect();
        const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
        const ny = 1 - ((clientY - rect.top) / rect.height) * 2;
        const aspect = rect.width / rect.height;
        return {
          x: viewCenter.x + nx * (2.35 * aspect) / zoom,
          y: viewCenter.y + ny * 2.35 / zoom,
        };
      };

      const onPointerDown = (event: PointerEvent) => {
        if (event.button === 1 || event.shiftKey) {
          dragging = true;
          lastPointer = { x: event.clientX, y: event.clientY };
          canvas.setPointerCapture(event.pointerId);
        }
      };
      const onPointerUp = () => { dragging = false; };
      const onPointerMove = (event: PointerEvent) => {
        if (dragging) {
          const rect = canvas.getBoundingClientRect();
          const dx = (event.clientX - lastPointer.x) / rect.width;
          const dy = (event.clientY - lastPointer.y) / rect.height;
          viewCenter.x -= dx * 4.7 * (rect.width / rect.height) / zoom;
          viewCenter.y += dy * 4.7 / zoom;
          lastPointer = { x: event.clientX, y: event.clientY };
          clearAccumulation();
          return;
        }
        const next = toComplex(event.clientX, event.clientY);
        pointRef.current = next;
        const now = performance.now();
        if (now - lastSpawnTime >= 120) {
          spawnCloud(next);
          lastSpawnTime = now;
        }
        setPoint(next);
      };
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        const before = toComplex(event.clientX, event.clientY);
        zoom = Math.max(0.55, Math.min(70, zoom * Math.exp(-event.deltaY * 0.001)));
        const after = toComplex(event.clientX, event.clientY);
        viewCenter.x += before.x - after.x;
        viewCenter.y += before.y - after.y;
        clearAccumulation();
      };
      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointercancel", onPointerUp);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      ro = new ResizeObserver(() => resize());
      ro.observe(canvas);
      resize();
      clearAccumulation();
      resetStreams();

      const onVisibilityChange = () => {
        pageVisible = !document.hidden;
        if (!pageVisible && frame) {
          cancelAnimationFrame(frame);
          frame = 0;
        } else if (pageVisible && !frame && !disposed) {
          lastFrameTime = performance.now();
          smoothFrameMs = 16.7;
          frame = requestAnimationFrame(draw);
        }
      };
      document.addEventListener("visibilitychange", onVisibilityChange);

      engineRef.current = {
        update(next) {
          if (next.iterations !== undefined) {
            resetStreams();
            clearAccumulation();
          }
        },
        reset() {
          viewCenter = { x: -0.62, y: 0 };
          zoom = DEFAULT_ZOOM;
          resetStreams();
          clearAccumulation();
        },
      };

      function draw(now: number) {
        frame = 0;
        if (disposed || !pageVisible || !textures.length) return;
        const delta = now - lastFrameTime;
        lastFrameTime = now;
        smoothFrameMs = smoothFrameMs * 0.92 + delta * 0.08;

        const cfg = settingsRef.current;
        const batchIterations = Math.max(1, Math.floor(FRAME_POINT_BUDGET / Math.max(currentParticles, 1)));
        const aspect = textureSize.width / textureSize.height;
        const point = pointRef.current;
        const params = new Float32Array(16);
        params[0] = point.x;
        params[1] = point.y;
        params[2] = aspect;
        params[3] = zoom;
        new Uint32Array(params.buffer)[4] = currentParticles;
        new Uint32Array(params.buffer)[5] = cfg.iterations;
        new Uint32Array(params.buffer)[6] = batchIterations;
        new Uint32Array(params.buffer)[7] = 0;
        params[8] = 0;
        params[9] = 0;
        params[10] = viewCenter.x;
        params[11] = viewCenter.y;
        device.queue.writeBuffer(paramsBuffer, 0, params);

        const alpha = 0.24;
        device.queue.writeBuffer(orbitStyleBuffer, 0, new Float32Array([
          alpha,
          (now * 0.00002) % 0.15,
          0, 0,
        ]));
        const fade = 0.8 + cfg.persistence * 0.002;
        device.queue.writeBuffer(fadeBuffer, 0, new Float32Array([fade, 1.8, 0, 0]));

        const destination = textures[1 - textureIndex];
        const encoder = device.createCommandEncoder();
        const compute = encoder.beginComputePass();
        compute.setPipeline(computePipeline);
        compute.setBindGroup(0, computeBindGroup);
        compute.dispatchWorkgroups(Math.ceil(currentParticles / WORKGROUP_SIZE));
        compute.end();

        const fadePass = encoder.beginRenderPass({
          colorAttachments: [{ view: destination.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
        });
        fadePass.setPipeline(fadePipeline);
        fadePass.setBindGroup(0, fadeBindGroups[textureIndex]);
        fadePass.draw(3);
        fadePass.end();

        const orbitPass = encoder.beginRenderPass({
          colorAttachments: [{ view: destination.createView(), loadOp: "load", storeOp: "store" }],
        });
        orbitPass.setVertexBuffer(0, vertexBuffer);
        const pointCount = currentParticles * batchIterations;
        orbitPass.setPipeline(orbitPipeline);
        orbitPass.setBindGroup(0, orbitBindGroup);
        orbitPass.draw(pointCount);
        orbitPass.end();

        const displayPass = encoder.beginRenderPass({
          colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        });
        displayPass.setPipeline(displayPipeline);
        displayPass.setBindGroup(0, displayBindGroups[1 - textureIndex]);
        displayPass.draw(3);
        displayPass.end();
        device.queue.submit([encoder.finish()]);
        textureIndex = 1 - textureIndex;

        if (now - lastStatTime > 500) {
          const fps = 1000 / smoothFrameMs;
          const samples = currentParticles * batchIterations;
          setStats({ fps, samples, throughput: samples * fps, particles: currentParticles });
          lastStatTime = now;
        }
        if (pageVisible) frame = requestAnimationFrame(draw);
      }

      if (pageVisible) frame = requestAnimationFrame(draw);
      device.lost.then(() => {
        if (!disposed) setError("The GPU connection was lost. Reload the page to restart the engine.");
      });

      return () => {
        document.removeEventListener("visibilitychange", onVisibilityChange);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("wheel", onWheel);
        textures.forEach((texture) => texture.destroy());
        vertexBuffer.destroy();
        stateBuffer.destroy();
        paramsBuffer.destroy();
        orbitStyleBuffer.destroy();
        fadeBuffer.destroy();
        device.destroy();
      };
    }

    let cleanup: (() => void) | undefined;
    start().then((fn) => { cleanup = fn; }).catch((reason) => setError(reason instanceof Error ? reason.message : "The GPU engine could not start."));
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      ro?.disconnect();
      cleanup?.();
      engineRef.current = null;
    };
  }, []);

  return (
    <main className="appShell">
      <header className="topbar">
        <div className="brand" aria-label="Orbit Lab">
          <span className="brandMark" aria-hidden="true" />
          <span>
            <span className="brandName">Orbit Lab</span><br />
            <span className="brandEdition">Mandelbrot trajectory engine</span>
          </span>
        </div>
        <p className="topCopy">Trace z² + c at GPU speed. Each cursor sample launches a compact cloud of nearby orbits—the first step from Mandelbrot to Buddhabrot.</p>
      </header>

      <div className="orbitWorkspace">
        <section className="lab orbitCanvasPanel" aria-label="Interactive Mandelbrot orbit field">
          <canvas ref={canvasRef} className="canvas" aria-label="GPU rendered complex plane. Move the pointer to change c." />
          <div className="gridOverlay" aria-hidden="true" />
          {error && (
            <div className="fallback" role="alert">
              <div className="fallbackCard">
                <h2>WebGPU needed.</h2>
                <p>{error}</p>
              </div>
            </div>
          )}
        </section>

        <div className="orbitSidebar">

        <aside className="panel controls">
          <div className="panelEyebrow"><span className="liveDot" /> Live complex coordinate</div>
          <p className="coordinate">
            {point.x.toFixed(5)} {point.y < 0 ? "−" : "+"} {Math.abs(point.y).toFixed(5)}i
          </p>
          <p className="pointState">{escapeLabel(point.x, point.y, settings.iterations)}</p>

          <div className="stat"><span className="statLabel">Cursor cloud</span><span className="statValue">{CURSOR_SEEDS} random seeds</span></div>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Orbit depth</span>
              <span className="controlValue">{settings.iterations.toLocaleString()} total · {formatCount(FRAME_POINT_BUDGET)}/frame</span>
            </span>
            <input className="range" type="range" min="5" max={Math.log2(MAX_ITERATIONS)} step="0.125" value={Math.log2(settings.iterations)} onChange={(event) => patchSettings({ iterations: Math.min(MAX_ITERATIONS, Math.round(2 ** Number(event.target.value))) })} />
          </label>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Trail memory</span>
              <span className="controlValue">{settings.persistence}%</span>
            </span>
            <input className="range" type="range" min="0" max="99" step="1" value={settings.persistence} onChange={(event) => patchSettings({ persistence: Number(event.target.value) })} />
          </label>

          <div className="buttonRow singleButton">
            <button className="button" onClick={() => engineRef.current?.reset()}>Reset view</button>
          </div>
        </aside>

        <aside className="panel stats" aria-label="Live performance statistics">
          <div className="stat"><span className="statLabel">Frame rate</span><span className="statValue">{stats.fps.toFixed(0)} fps</span></div>
          <div className="stat"><span className="statLabel">Per frame</span><span className="statValue">{formatCount(stats.samples)}</span></div>
          <div className="stat"><span className="statLabel">Per second</span><span className="statValue">{formatCount(stats.throughput)}</span></div>
          <div className="stat"><span className="statLabel">Orbit streams</span><span className="statValue">{stats.particles}</span></div>
        </aside>

        <p className="hint">Move: launch 100 random seeds · Scroll: zoom · Shift + drag: pan</p>

        </div>
      </div>
    </main>
  );
}

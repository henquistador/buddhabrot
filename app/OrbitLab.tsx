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
  density: number;
  persistence: number;
  pointSize: number;
  sizeSlope: number;
  halo: boolean;
  targetFps: number;
  adaptive: boolean;
};

const MAX_PARTICLES = 16384;
const MAX_ITERATIONS = 1_048_576;
const FRAME_BATCH = 256;
const MAX_FRAME_POINTS = MAX_PARTICLES * FRAME_BATCH;
const WORKGROUP_SIZE = 64;

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

fn hash(v: u32) -> f32 {
  var x = v;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = (x >> 16u) ^ x;
  return f32(x) / 4294967295.0;
}

fn toClip(z: vec2f) -> vec2f {
  let view = vec2f(2.35 * params.aspect, 2.35) / params.zoom;
  return (z - params.center) / view;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let particle = id.x;
  if (particle >= params.particleCount) { return; }

  var state = states[particle];
  let newGeneration = state.generation != params.generation;
  let needsRestart = newGeneration || state.alive == 0u || state.step >= params.iterations;
  if (needsRestart) {
    state.cycle = select(state.cycle + 1u, 0u, newGeneration);
    let seed = particle * 0x9e3779b9u ^ params.generation * 0x85ebca6bu ^ state.cycle * 0xc2b2ae35u;
    let jitter = vec2f(hash(seed), hash(seed ^ 0x68bc21ebu)) * 2.0 - 1.0;
    state.z = vec2f(0.0);
    state.sampleC = params.c + jitter * params.spread;
    state.step = 0u;
    state.generation = params.generation;
    state.alive = 1u;
  }

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
  iterations: f32,
  pointSize: f32,
  resolution: vec2f,
  sizeSlope: f32,
  pad: f32,
}
@group(0) @binding(0) var<uniform> style: OrbitUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
}

@vertex
fn vs(@location(0) position: vec2f, @location(1) stepT: f32) -> VSOut {
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

const quadOrbitShader = /* wgsl */ `
struct OrbitUniforms {
  alpha: f32,
  hue: f32,
  iterations: f32,
  pointSize: f32,
  resolution: vec2f,
  sizeSlope: f32,
  pad: f32,
}
@group(0) @binding(0) var<uniform> style: OrbitUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) local: vec2f,
}

@vertex
fn vs(
  @location(0) center: vec2f,
  @location(1) stepT: f32,
  @builtin(vertex_index) vertexIndex: u32,
) -> VSOut {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let diameter = clamp(style.pointSize * exp2(stepT * style.sizeSlope), 0.2, 24.0);
  let clipOffset = corner * diameter / style.resolution;
  var out: VSOut;
  out.position = vec4f(center + clipOffset, 0.0, 1.0);
  out.color = mix(vec3f(0.20, 1.0, 0.60), vec3f(0.30, 0.48, 1.0), stepT + style.hue);
  out.local = corner;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let radius2 = dot(in.local, in.local);
  if (radius2 > 1.0) { discard; }
  let coverage = 1.0 - smoothstep(0.62, 1.0, radius2);
  return vec4f(in.color * style.alpha * coverage, style.alpha * coverage);
}
`;

const fullscreenShader = /* wgsl */ `
struct FadeUniforms { fade: f32, gain: f32, halo: f32, pad: f32 }
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
  var raw = center * settings.gain;
  if (settings.halo > 0.5) {
    let texel = 1.0 / vec2f(textureDimensions(previous));
    let neighbors = (
      textureSample(previous, previousSampler, in.uv + vec2f(texel.x, 0.0)).rgb +
      textureSample(previous, previousSampler, in.uv - vec2f(texel.x, 0.0)).rgb +
      textureSample(previous, previousSampler, in.uv + vec2f(0.0, texel.y)).rgb +
      textureSample(previous, previousSampler, in.uv - vec2f(0.0, texel.y)).rgb
    ) * 0.25;
    raw = (center + neighbors * 0.48) * settings.gain;
  }
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
    iterations: 32768,
    density: 13,
    persistence: 94,
    pointSize: 0.65,
    sizeSlope: -3,
    halo: false,
    targetFps: 60,
    adaptive: true,
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
        size: MAX_PARTICLES * 32,
        usage: usage.STORAGE,
      });
      const paramsBuffer = device.createBuffer({ size: 64, usage: usage.UNIFORM | usage.COPY_DST });
      const orbitStyleBuffer = device.createBuffer({ size: 32, usage: usage.UNIFORM | usage.COPY_DST });
      const fadeBuffer = device.createBuffer({ size: 16, usage: usage.UNIFORM | usage.COPY_DST });
      const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

      const computeModule = device.createShaderModule({ code: computeShader });
      const orbitModule = device.createShaderModule({ code: orbitShader });
      const quadOrbitModule = device.createShaderModule({ code: quadOrbitShader });
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
      const quadOrbitPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: quadOrbitModule,
          entryPoint: "vs",
          buffers: [{
            arrayStride: 16,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x2" },
              { shaderLocation: 1, offset: 8, format: "float32" },
            ],
          }],
        },
        fragment: {
          module: quadOrbitModule,
          entryPoint: "fs",
          targets: [{
            format: "rgba16float",
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          }],
        },
        primitive: { topology: "triangle-list" },
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
      const quadOrbitBindGroup = device.createBindGroup({
        layout: quadOrbitPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: orbitStyleBuffer } }],
      });

      let textures: any[] = [];
      let fadeBindGroups: any[] = [];
      let displayBindGroups: any[] = [];
      let textureSize = { width: 0, height: 0 };
      let pixelRatio = 1;
      let textureIndex = 0;
      let currentParticles = Math.min(MAX_PARTICLES, 2 ** settingsRef.current.density);
      let viewCenter = { x: -0.62, y: 0 };
      let zoom = 1;
      let generation = 1;
      let dragging = false;
      let lastPointer = { x: 0, y: 0 };
      let frameCount = 0;
      let lastStatTime = performance.now();
      let lastFrameTime = performance.now();
      let smoothFrameMs = 16.7;

      function bumpGeneration() {
        generation = generation === 0xffffffff ? 1 : generation + 1;
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
        // Supersample low-DPI displays so a one-device-pixel point stays visually tiny.
        const dpr = Math.max(1.35, Math.min(window.devicePixelRatio || 1, 2));
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width * dpr));
        const height = Math.max(1, Math.round(rect.height * dpr));
        pixelRatio = dpr;
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
          bumpGeneration();
          clearAccumulation();
          return;
        }
        const next = toComplex(event.clientX, event.clientY);
        pointRef.current = next;
        bumpGeneration();
        setPoint(next);
      };
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        const before = toComplex(event.clientX, event.clientY);
        zoom = Math.max(0.55, Math.min(70, zoom * Math.exp(-event.deltaY * 0.001)));
        const after = toComplex(event.clientX, event.clientY);
        viewCenter.x += before.x - after.x;
        viewCenter.y += before.y - after.y;
        bumpGeneration();
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

      engineRef.current = {
        update(next) {
          if (typeof next.density === "number") currentParticles = Math.min(MAX_PARTICLES, 2 ** next.density);
          if ((next.pointSize !== undefined && next.pointSize > 1) || (next.sizeSlope !== undefined && Math.abs(next.sizeSlope) > 0.001)) {
            currentParticles = Math.max(512, Math.floor(currentParticles / 4 / 64) * 64);
          }
          if (next.iterations !== undefined || next.density !== undefined) bumpGeneration();
          if (next.persistence !== undefined || next.iterations !== undefined || next.pointSize !== undefined || next.sizeSlope !== undefined || next.halo !== undefined) clearAccumulation();
        },
        reset() {
          viewCenter = { x: -0.62, y: 0 };
          zoom = 1;
          bumpGeneration();
          clearAccumulation();
        },
      };

      function draw(now: number) {
        if (disposed || !textures.length) return;
        const delta = now - lastFrameTime;
        lastFrameTime = now;
        smoothFrameMs = smoothFrameMs * 0.92 + delta * 0.08;
        frameCount++;

        const cfg = settingsRef.current;
        const particleCap = Math.min(MAX_PARTICLES, 2 ** cfg.density);
        currentParticles = Math.min(currentParticles, particleCap);
        if (cfg.adaptive && frameCount % 30 === 0) {
          const targetMs = 1000 / cfg.targetFps;
          if (smoothFrameMs > targetMs * 1.12) currentParticles = Math.max(512, Math.floor(currentParticles * 0.82 / 64) * 64);
          else if (smoothFrameMs < targetMs * 0.82) currentParticles = Math.min(particleCap, Math.ceil(currentParticles * 1.12 / 64) * 64);
        }

        const aspect = textureSize.width / textureSize.height;
        const point = pointRef.current;
        const params = new Float32Array(16);
        params[0] = point.x;
        params[1] = point.y;
        params[2] = aspect;
        params[3] = zoom;
        new Uint32Array(params.buffer)[4] = currentParticles;
        new Uint32Array(params.buffer)[5] = cfg.iterations;
        new Uint32Array(params.buffer)[6] = FRAME_BATCH;
        new Uint32Array(params.buffer)[7] = generation;
        params[8] = 0.014 / Math.sqrt(zoom);
        params[9] = 0;
        params[10] = viewCenter.x;
        params[11] = viewCenter.y;
        device.queue.writeBuffer(paramsBuffer, 0, params);

        const baseAlpha = Math.min(0.026, 1.05 / Math.sqrt(currentParticles));
        const sizeAlpha = cfg.pointSize <= 1 ? cfg.pointSize * cfg.pointSize : 1 / Math.sqrt(cfg.pointSize);
        const alpha = baseAlpha * sizeAlpha;
        device.queue.writeBuffer(orbitStyleBuffer, 0, new Float32Array([
          alpha,
          (now * 0.00002) % 0.15,
          cfg.iterations,
          cfg.pointSize * pixelRatio,
          textureSize.width,
          textureSize.height,
          cfg.sizeSlope,
          0,
        ]));
        const fade = 0.84 + cfg.persistence * 0.00165;
        const gain = 1.5;
        device.queue.writeBuffer(fadeBuffer, 0, new Float32Array([fade, gain, cfg.halo ? 1 : 0, 0]));

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
        const pointCount = currentParticles * FRAME_BATCH;
        if (cfg.pointSize <= 1 && Math.abs(cfg.sizeSlope) < 0.001) {
          orbitPass.setPipeline(orbitPipeline);
          orbitPass.setBindGroup(0, orbitBindGroup);
          orbitPass.draw(pointCount);
        } else {
          orbitPass.setPipeline(quadOrbitPipeline);
          orbitPass.setBindGroup(0, quadOrbitBindGroup);
          orbitPass.draw(6, pointCount);
        }
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
          const samples = currentParticles * FRAME_BATCH;
          setStats({ fps, samples, throughput: samples * fps, particles: currentParticles });
          lastStatTime = now;
        }
        frame = requestAnimationFrame(draw);
      }

      frame = requestAnimationFrame(draw);
      device.lost.then(() => {
        if (!disposed) setError("The GPU connection was lost. Reload the page to restart the engine.");
      });

      return () => {
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
        <p className="topCopy">Trace z² + c at GPU speed. Each glow is a stack of nearby orbits—the first step from Mandelbrot to Buddhabrot.</p>
      </header>

      <section className="lab" aria-label="Interactive Mandelbrot orbit field">
        <canvas ref={canvasRef} className="canvas" aria-label="GPU rendered complex plane. Move the pointer to change c." />
        <div className="gridOverlay" aria-hidden="true" />

        <aside className="panel controls">
          <div className="panelEyebrow"><span className="liveDot" /> Live complex coordinate</div>
          <p className="coordinate">
            {point.x.toFixed(5)} {point.y < 0 ? "−" : "+"} {Math.abs(point.y).toFixed(5)}i
          </p>
          <p className="pointState">{escapeLabel(point.x, point.y, settings.iterations)}</p>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Orbit depth</span>
              <span className="controlValue">{settings.iterations.toLocaleString()} total · {FRAME_BATCH}/frame</span>
            </span>
            <input className="range" type="range" min="5" max="20" step="1" value={Math.log2(settings.iterations)} onChange={(event) => patchSettings({ iterations: 2 ** Number(event.target.value) })} />
          </label>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Seed field</span>
              <span className="controlValue">{formatCount(2 ** settings.density)} max samples</span>
            </span>
            <input className="range" type="range" min="9" max="14" step="1" value={settings.density} onChange={(event) => patchSettings({ density: Number(event.target.value), adaptive: false })} />
          </label>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Trail memory</span>
              <span className="controlValue">{settings.persistence}%</span>
            </span>
            <input className="range" type="range" min="4" max="99" step="1" value={settings.persistence} onChange={(event) => patchSettings({ persistence: Number(event.target.value) })} />
          </label>

          <div className="miniControlGrid">
            <label className="controlRow miniControl">
              <span className="controlHead">
                <span className="controlLabel">Point size</span>
                <span className="controlValue">{settings.pointSize.toFixed(2)} px</span>
              </span>
              <input className="range" type="range" min="0.25" max="4" step="0.05" value={settings.pointSize} onChange={(event) => patchSettings({ pointSize: Number(event.target.value) })} />
            </label>

            <label className="controlRow miniControl">
              <span className="controlHead">
                <span className="controlLabel">Size slope</span>
                <span className="controlValue">{settings.sizeSlope > 0 ? "+" : ""}{settings.sizeSlope.toFixed(2)}</span>
              </span>
              <input className="range" type="range" min="-3" max="3" step="0.1" value={settings.sizeSlope} onChange={(event) => patchSettings({ sizeSlope: Number(event.target.value) })} />
            </label>
          </div>

          <label className="toggleRow">
            <input type="checkbox" checked={settings.halo} onChange={(event) => patchSettings({ halo: event.target.checked })} />
            <span>Halo</span>
            <span className="controlValue">{settings.halo ? "On" : "Off"}</span>
          </label>

          <div className="buttonRow">
            <button className={`button ${settings.adaptive ? "buttonPrimary" : ""}`} onClick={() => patchSettings({ adaptive: !settings.adaptive })} aria-pressed={settings.adaptive}>
              {settings.adaptive ? `Auto · ${settings.targetFps} FPS` : "Enable auto-load"}
            </button>
            <button className="button" onClick={() => engineRef.current?.reset()}>Reset view</button>
          </div>
        </aside>

        <aside className="panel stats" aria-label="Live performance statistics">
          <div className="stat"><span className="statLabel">Frame rate</span><span className="statValue">{stats.fps.toFixed(0)} fps</span></div>
          <div className="stat"><span className="statLabel">Per frame</span><span className="statValue">{formatCount(stats.samples)}</span></div>
          <div className="stat"><span className="statLabel">Per second</span><span className="statValue">{formatCount(stats.throughput)}</span></div>
          <div className="stat"><span className="statLabel">Seeds</span><span className="statValue">{formatCount(stats.particles)}</span></div>
        </aside>

        <p className="hint">Move: choose c · Scroll: zoom · Shift + drag: pan</p>

        {error && (
          <div className="fallback" role="alert">
            <div className="fallbackCard">
              <h2>WebGPU needed.</h2>
              <p>{error}</p>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

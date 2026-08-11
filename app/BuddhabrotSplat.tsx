"use client";

/* WebGPU types are not part of this project's TypeScript DOM library yet. */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { useCallback, useEffect, useRef, useState } from "react";

type Stats = {
  fps: number;
  splats: number;
  throughput: number;
  seeds: number;
};

type EngineSettings = {
  iterations: number;
  density: number;
  persistence: number;
  splatScale: number;
  stretch: number;
  volumeDepth: number;
  seedRadius: number;
  bloom: boolean;
  autoRotate: boolean;
  targetFps: number;
  adaptive: boolean;
};

const MAX_SEEDS = 8192;
const MAX_SPLATS = 2_097_152;
const WORKGROUP_SIZE = 64;
const INITIAL_SETTINGS: EngineSettings = {
  iterations: 256,
  density: 12,
  persistence: 80,
  splatScale: 1,
  stretch: 0.72,
  volumeDepth: 2.2,
  seedRadius: 1.72,
  bloom: true,
  autoRotate: true,
  targetFps: 60,
  adaptive: true,
};

// Each candidate c is iterated once to determine whether it escapes. Only then
// is its orbit written. The third coordinate is normalized orbit time, turning
// the usual 2D Buddhabrot exposure into a volume of trajectories.
const computeShader = /* wgsl */ `
struct Params {
  focus: vec2f,
  spread: f32,
  depth: f32,
  seedCount: u32,
  iterations: u32,
  epoch: u32,
  minOrbit: u32,
}

struct Splat {
  center: vec4f,
  tangent: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> splats: array<Splat>;

fn hash(v: u32) -> f32 {
  var x = v;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = (x >> 16u) ^ x;
  return f32(x) / 4294967295.0;
}

fn iterate(z: vec2f, c: vec2f) -> vec2f {
  return vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
}

fn isKnownInterior(c: vec2f) -> bool {
  let bulb = (c.x + 1.0) * (c.x + 1.0) + c.y * c.y;
  let q = (c.x - 0.25) * (c.x - 0.25) + c.y * c.y;
  return bulb <= 0.0625 || q * (q + c.x - 0.25) <= 0.25 * c.y * c.y;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let seed = id.x;
  if (seed >= params.seedCount) { return; }

  let randomBase = seed * 747796405u + params.epoch * 2891336453u;
  let jitter = vec2f(hash(randomBase), hash(randomBase ^ 0x9e3779b9u)) * 2.0 - 1.0;
  let c = params.focus + jitter * params.spread;
  var escapeAt = 0u;
  var z = vec2f(0.0);

  if (!isKnownInterior(c)) {
    for (var step = 0u; step < params.iterations; step++) {
      z = iterate(z, c);
      if (dot(z, z) > 4.0) {
        escapeAt = step + 1u;
        break;
      }
    }
  }

  z = vec2f(0.0);
  for (var step = 0u; step < params.iterations; step++) {
    let slot = seed * params.iterations + step;
    if (escapeAt < params.minOrbit || step >= escapeAt) {
      splats[slot].center = vec4f(0.0, 0.0, 0.0, 0.0);
      splats[slot].tangent = vec4f(0.0);
      continue;
    }

    let next = iterate(z, c);
    let after = iterate(next, c);
    let denom = max(f32(escapeAt - 1u), 1.0);
    let t = f32(step) / denom;
    let layer = (t - 0.5) * params.depth;
    let dz = params.depth / denom;
    splats[slot].center = vec4f(next.x, next.y, layer, 1.0);
    splats[slot].tangent = vec4f(after - next, dz, t);
    z = next;
  }
}
`;

const splatShader = /* wgsl */ `
struct Style {
  alpha: f32,
  scale: f32,
  stretch: f32,
  sizeSlope: f32,
  resolution: vec2f,
  yaw: f32,
  pitch: f32,
  distance: f32,
  aspect: f32,
  focal: f32,
  shimmer: f32,
  pad: vec4f,
}

@group(0) @binding(0) var<uniform> style: Style;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) color: vec3f,
}

fn viewPoint(p: vec3f) -> vec3f {
  let cy = cos(style.yaw);
  let sy = sin(style.yaw);
  let cp = cos(style.pitch);
  let sp = sin(style.pitch);
  let yawed = vec3f(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
  return vec3f(yawed.x, cp * yawed.y - sp * yawed.z,
               sp * yawed.y + cp * yawed.z + style.distance);
}

fn project(p: vec3f) -> vec2f {
  return p.xy / (max(p.z, 0.05) * style.focal * vec2f(style.aspect, 1.0));
}

@vertex
fn vs(
  @location(0) centerData: vec4f,
  @location(1) tangentData: vec4f,
  @builtin(vertex_index) vertexIndex: u32,
) -> VSOut {
  let corners = array<vec2f, 6>(
    vec2f(-3.0, -3.0), vec2f(3.0, -3.0), vec2f(-3.0, 3.0),
    vec2f(-3.0, 3.0), vec2f(3.0, -3.0), vec2f(3.0, 3.0)
  );
  var out: VSOut;
  if (centerData.w < 0.5) {
    out.position = vec4f(2.0, 2.0, 0.0, 1.0);
    out.local = vec2f(99.0);
    out.color = vec3f(0.0);
    return out;
  }

  let centerView = viewPoint(centerData.xyz);
  let tangentView = viewPoint(centerData.xyz + tangentData.xyz) - centerView;
  let centerNdc = project(centerView);
  let tangentNdc = project(centerView + tangentView) - centerNdc;
  let tangentPx = tangentNdc * style.resolution;
  let tangentLength = max(length(tangentPx), 0.0001);
  let majorDir = tangentPx / tangentLength;
  let minorDir = vec2f(-majorDir.y, majorDir.x);

  let perspective = style.resolution.y / max(centerView.z * style.focal, 0.1);
  let sigma = clamp(0.012 * style.scale * perspective * exp2(tangentData.w * style.sizeSlope), 0.32, 18.0);
  let major = sigma * mix(1.0, 3.6, style.stretch);
  let minor = sigma * mix(1.0, 0.48, style.stretch);
  let corner = corners[vertexIndex];
  let pixelOffset = majorDir * corner.x * major + minorDir * corner.y * minor;
  let ndcOffset = pixelOffset * 2.0 / style.resolution;

  out.position = vec4f((centerNdc + ndcOffset) * centerView.z, 0.0, centerView.z);
  out.local = corner;
  let early = vec3f(0.12, 0.93, 0.78);
  let middle = vec3f(0.34, 0.48, 1.0);
  let late = vec3f(1.0, 0.32, 0.66);
  let t = tangentData.w;
  let base = select(mix(early, middle, t * 2.0), mix(middle, late, (t - 0.5) * 2.0), t > 0.5);
  out.color = base * (0.86 + 0.14 * cos(style.shimmer + t * 8.0));
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let radius2 = dot(in.local, in.local);
  if (radius2 > 9.0) { discard; }
  let gaussian = exp(-0.5 * radius2);
  let energy = gaussian * style.alpha;
  return vec4f(in.color * energy, energy);
}
`;

const fullscreenShader = /* wgsl */ `
struct Display { fade: f32, gain: f32, bloom: f32, pad: f32 }
@group(0) @binding(0) var previous: texture_2d<f32>;
@group(0) @binding(1) var previousSampler: sampler;
@group(0) @binding(2) var<uniform> display: Display;

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
  return textureSample(previous, previousSampler, in.uv) * display.fade;
}

@fragment
fn displayFs(in: VSOut) -> @location(0) vec4f {
  let center = textureSample(previous, previousSampler, in.uv).rgb;
  var light = center;
  if (display.bloom > 0.5) {
    let texel = 2.0 / vec2f(textureDimensions(previous));
    light += 0.10 * (
      textureSample(previous, previousSampler, in.uv + vec2f(texel.x, 0.0)).rgb +
      textureSample(previous, previousSampler, in.uv - vec2f(texel.x, 0.0)).rgb +
      textureSample(previous, previousSampler, in.uv + vec2f(0.0, texel.y)).rgb +
      textureSample(previous, previousSampler, in.uv - vec2f(0.0, texel.y)).rgb
    );
  }
  let mapped = vec3f(1.0) - exp(-light * display.gain);
  let graded = pow(mapped, vec3f(0.86));
  return vec4f(vec3f(0.006, 0.008, 0.018) + graded, 1.0);
}
`;

function formatCount(value: number) {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(Math.round(value));
}

export default function BuddhabrotSplat() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<{ update: (next: Partial<EngineSettings>) => void; reset: () => void } | null>(null);
  const settingsRef = useRef<EngineSettings>({ ...INITIAL_SETTINGS });
  const [settings, setSettings] = useState<EngineSettings>({ ...INITIAL_SETTINGS });
  const [stats, setStats] = useState<Stats>({ fps: 0, splats: 0, throughput: 0, seeds: 0 });
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
        setError("WebGPU is required. Open this in current Chrome, Edge, or Safari with hardware acceleration enabled.");
        return;
      }

      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) {
        setError("No compatible GPU adapter was found. Check browser hardware acceleration.");
        return;
      }

      const device = await adapter.requestDevice();
      if (disposed) return;
      const context = canvas.getContext("webgpu") as any;
      const canvasFormat = gpu.getPreferredCanvasFormat();
      context.configure({ device, format: canvasFormat, alphaMode: "opaque" });

      const usage = (globalThis as any).GPUBufferUsage;
      const textureUsage = (globalThis as any).GPUTextureUsage;
      const splatBuffer = device.createBuffer({
        size: MAX_SPLATS * 32,
        usage: usage.STORAGE | usage.VERTEX,
      });
      const paramsBuffer = device.createBuffer({ size: 32, usage: usage.UNIFORM | usage.COPY_DST });
      const styleBuffer = device.createBuffer({ size: 64, usage: usage.UNIFORM | usage.COPY_DST });
      const displayBuffer = device.createBuffer({ size: 16, usage: usage.UNIFORM | usage.COPY_DST });
      const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

      const computeModule = device.createShaderModule({ code: computeShader });
      const splatModule = device.createShaderModule({ code: splatShader });
      const fullscreenModule = device.createShaderModule({ code: fullscreenShader });
      const computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: computeModule, entryPoint: "main" },
      });
      const splatPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: splatModule,
          entryPoint: "vs",
          buffers: [{
            arrayStride: 32,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x4" },
              { shaderLocation: 1, offset: 16, format: "float32x4" },
            ],
          }],
        },
        fragment: {
          module: splatModule,
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
          { binding: 1, resource: { buffer: splatBuffer } },
        ],
      });
      const splatBindGroup = device.createBindGroup({
        layout: splatPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: styleBuffer } }],
      });

      let textures: any[] = [];
      let fadeBindGroups: any[] = [];
      let displayBindGroups: any[] = [];
      let textureSize = { width: 0, height: 0 };
      let textureIndex = 0;
      let pixelRatio = 1;
      let currentSeeds = 2 ** settingsRef.current.density;
      let yaw = -0.52;
      let pitch = -0.24;
      let distance = 5.4;
      let dragging = false;
      let lastPointer = { x: 0, y: 0 };
      let frameCount = 0;
      let epoch = 1;
      let lastStatTime = performance.now();
      let lastFrameTime = performance.now();
      let smoothFrameMs = 16.7;

      function makeTextureBindGroup(pipeline: any, texture: any) {
        return device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: displayBuffer } },
          ],
        });
      }

      function resize() {
        if (!canvas) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
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
        clearAccumulation();
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

      const onPointerDown = (event: PointerEvent) => {
        if (event.button !== 0) return;
        dragging = true;
        lastPointer = { x: event.clientX, y: event.clientY };
        canvas.setPointerCapture(event.pointerId);
      };
      const onPointerUp = () => { dragging = false; };
      const onPointerMove = (event: PointerEvent) => {
        if (!dragging) return;
        yaw += (event.clientX - lastPointer.x) * 0.006;
        pitch = Math.max(-1.35, Math.min(1.35, pitch + (event.clientY - lastPointer.y) * 0.005));
        lastPointer = { x: event.clientX, y: event.clientY };
        clearAccumulation();
      };
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        distance = Math.max(2.5, Math.min(11, distance * Math.exp(event.deltaY * 0.001)));
        clearAccumulation();
      };
      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointercancel", onPointerUp);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      ro = new ResizeObserver(resize);
      ro.observe(canvas);
      resize();

      engineRef.current = {
        update(next) {
          const cfg = settingsRef.current;
          const cap = Math.max(64, Math.floor(MAX_SPLATS / cfg.iterations / 64) * 64);
          if (typeof next.density === "number" || typeof next.iterations === "number") {
            currentSeeds = Math.min(MAX_SEEDS, cap, 2 ** cfg.density);
          }
          clearAccumulation();
        },
        reset() {
          yaw = -0.52;
          pitch = -0.24;
          distance = 5.4;
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
        const depthCap = Math.max(64, Math.floor(MAX_SPLATS / cfg.iterations / 64) * 64);
        const seedCap = Math.min(MAX_SEEDS, 2 ** cfg.density, depthCap);
        currentSeeds = Math.min(currentSeeds, seedCap);
        if (cfg.adaptive && frameCount % 30 === 0) {
          const targetMs = 1000 / cfg.targetFps;
          if (smoothFrameMs > targetMs * 1.12) currentSeeds = Math.max(256, Math.floor(currentSeeds * 0.82 / 64) * 64);
          else if (smoothFrameMs < targetMs * 0.82) currentSeeds = Math.min(seedCap, Math.ceil(currentSeeds * 1.12 / 64) * 64);
        }
        if (cfg.autoRotate && !dragging) yaw += delta * 0.000035;

        const params = new Float32Array(8);
        params[0] = -0.5;
        params[1] = 0;
        params[2] = cfg.seedRadius;
        params[3] = cfg.volumeDepth;
        const paramsU32 = new Uint32Array(params.buffer);
        paramsU32[4] = currentSeeds;
        paramsU32[5] = cfg.iterations;
        paramsU32[6] = epoch++;
        paramsU32[7] = 8;
        device.queue.writeBuffer(paramsBuffer, 0, params);

        const alpha = Math.min(0.045, 0.34 / Math.sqrt(currentSeeds));
        device.queue.writeBuffer(styleBuffer, 0, new Float32Array([
          alpha,
          cfg.splatScale * pixelRatio,
          cfg.stretch,
          0.7,
          textureSize.width,
          textureSize.height,
          yaw,
          pitch,
          distance,
          textureSize.width / textureSize.height,
          0.4663,
          now * 0.001,
          0, 0, 0, 0,
        ]));
        const fade = cfg.autoRotate ? 0.70 + cfg.persistence * 0.0026 : 0.76 + cfg.persistence * 0.00235;
        device.queue.writeBuffer(displayBuffer, 0, new Float32Array([Math.min(fade, 0.985), 1.35, cfg.bloom ? 1 : 0, 0]));

        const destination = textures[1 - textureIndex];
        const encoder = device.createCommandEncoder();
        const compute = encoder.beginComputePass();
        compute.setPipeline(computePipeline);
        compute.setBindGroup(0, computeBindGroup);
        compute.dispatchWorkgroups(Math.ceil(currentSeeds / WORKGROUP_SIZE));
        compute.end();

        const fadePass = encoder.beginRenderPass({
          colorAttachments: [{ view: destination.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
        });
        fadePass.setPipeline(fadePipeline);
        fadePass.setBindGroup(0, fadeBindGroups[textureIndex]);
        fadePass.draw(3);
        fadePass.end();

        const splatPass = encoder.beginRenderPass({
          colorAttachments: [{ view: destination.createView(), loadOp: "load", storeOp: "store" }],
        });
        splatPass.setPipeline(splatPipeline);
        splatPass.setBindGroup(0, splatBindGroup);
        splatPass.setVertexBuffer(0, splatBuffer);
        const splatSlots = currentSeeds * cfg.iterations;
        splatPass.draw(6, splatSlots);
        splatPass.end();

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
          setStats({ fps, splats: splatSlots, throughput: splatSlots * fps, seeds: currentSeeds });
          lastStatTime = now;
        }
        frame = requestAnimationFrame(draw);
      }

      frame = requestAnimationFrame(draw);
      device.lost.then(() => {
        if (!disposed) setError("The GPU connection was lost. Reload to restart the engine.");
      });

      return () => {
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("wheel", onWheel);
        textures.forEach((texture) => texture.destroy());
        splatBuffer.destroy();
        paramsBuffer.destroy();
        styleBuffer.destroy();
        displayBuffer.destroy();
        device.destroy();
      };
    }

    let cleanup: (() => void) | undefined;
    start().then((fn) => { cleanup = fn; }).catch((reason) => {
      setError(reason instanceof Error ? reason.message : "The GPU engine could not start.");
    });
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
        <div className="brand" aria-label="Buddhabrot Splat Lab">
          <span className="brandMark splatMark" aria-hidden="true" />
          <span>
            <span className="brandName">Buddhabrot Splat Lab</span><br />
            <span className="brandEdition">Escape-orbit volume · WebGPU</span>
          </span>
        </div>
        <p className="topCopy">Only escaping z² + c paths survive. Orbit time becomes depth; every step becomes an anisotropic Gaussian.</p>
      </header>

      <section className="lab" aria-label="Interactive 3D Buddhabrot Gaussian splat">
        <canvas ref={canvasRef} className="canvas" aria-label="GPU rendered 3D Buddhabrot. Drag to orbit and scroll to zoom." />
        <div className="vignette" aria-hidden="true" />

        <aside className="panel controls">
          <div className="panelEyebrow"><span className="liveDot" /> Live rejection sampler</div>
          <p className="coordinate">z = z² + c</p>
          <p className="pointState">ESCAPE-ONLY · XYT VOLUME · GAUSSIAN KERNEL</p>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Orbit depth</span>
              <span className="controlValue">{settings.iterations} iterations</span>
            </span>
            <input className="range" type="range" min="5" max="11" step="1" value={Math.log2(settings.iterations)} onChange={(event) => patchSettings({ iterations: 2 ** Number(event.target.value) })} />
          </label>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Candidate seeds</span>
              <span className="controlValue">{formatCount(2 ** settings.density)} max / frame</span>
            </span>
            <input className="range" type="range" min="8" max="13" step="1" value={settings.density} onChange={(event) => patchSettings({ density: Number(event.target.value), adaptive: false })} />
          </label>

          <div className="miniControlGrid">
            <label className="controlRow miniControl">
              <span className="controlHead">
                <span className="controlLabel">Volume depth</span>
                <span className="controlValue">{settings.volumeDepth.toFixed(1)}×</span>
              </span>
              <input className="range" type="range" min="0" max="4" step="0.1" value={settings.volumeDepth} onChange={(event) => patchSettings({ volumeDepth: Number(event.target.value) })} />
            </label>
            <label className="controlRow miniControl">
              <span className="controlHead">
                <span className="controlLabel">Seed radius</span>
                <span className="controlValue">{settings.seedRadius.toFixed(2)}</span>
              </span>
              <input className="range" type="range" min="0.8" max="2.2" step="0.02" value={settings.seedRadius} onChange={(event) => patchSettings({ seedRadius: Number(event.target.value) })} />
            </label>
          </div>

          <div className="miniControlGrid">
            <label className="controlRow miniControl">
              <span className="controlHead">
                <span className="controlLabel">Splat scale</span>
                <span className="controlValue">{settings.splatScale.toFixed(2)}×</span>
              </span>
              <input className="range" type="range" min="0.25" max="3" step="0.05" value={settings.splatScale} onChange={(event) => patchSettings({ splatScale: Number(event.target.value) })} />
            </label>
            <label className="controlRow miniControl">
              <span className="controlHead">
                <span className="controlLabel">Tangent stretch</span>
                <span className="controlValue">{Math.round(settings.stretch * 100)}%</span>
              </span>
              <input className="range" type="range" min="0" max="1" step="0.02" value={settings.stretch} onChange={(event) => patchSettings({ stretch: Number(event.target.value) })} />
            </label>
          </div>

          <label className="controlRow">
            <span className="controlHead">
              <span className="controlLabel">Exposure memory</span>
              <span className="controlValue">{settings.persistence}%</span>
            </span>
            <input className="range" type="range" min="0" max="98" step="1" value={settings.persistence} onChange={(event) => patchSettings({ persistence: Number(event.target.value) })} />
          </label>

          <div className="toggleGrid">
            <label className="toggleRow">
              <input type="checkbox" checked={settings.bloom} onChange={(event) => patchSettings({ bloom: event.target.checked })} />
              <span>Bloom</span>
              <span className="controlValue">{settings.bloom ? "On" : "Off"}</span>
            </label>
            <label className="toggleRow">
              <input type="checkbox" checked={settings.autoRotate} onChange={(event) => patchSettings({ autoRotate: event.target.checked })} />
              <span>Auto orbit</span>
              <span className="controlValue">{settings.autoRotate ? "On" : "Off"}</span>
            </label>
          </div>

          <div className="buttonRow">
            <button className={`button ${settings.adaptive ? "buttonPrimary" : ""}`} onClick={() => patchSettings({ adaptive: !settings.adaptive })} aria-pressed={settings.adaptive}>
              {settings.adaptive ? `Auto · ${settings.targetFps} FPS` : "Enable auto-load"}
            </button>
            <button className="button" onClick={() => engineRef.current?.reset()}>Reset camera</button>
          </div>
        </aside>

        <aside className="panel stats" aria-label="Live performance statistics">
          <div className="stat"><span className="statLabel">Frame rate</span><span className="statValue">{stats.fps.toFixed(0)} fps</span></div>
          <div className="stat"><span className="statLabel">Splat slots</span><span className="statValue">{formatCount(stats.splats)}</span></div>
          <div className="stat"><span className="statLabel">Slots / sec</span><span className="statValue">{formatCount(stats.throughput)}</span></div>
          <div className="stat"><span className="statLabel">Seeds tested</span><span className="statValue">{formatCount(stats.seeds)}</span></div>
        </aside>

        <p className="hint">Drag: orbit volume · Scroll: dolly · Escaping paths only</p>

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

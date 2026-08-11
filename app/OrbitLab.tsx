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
  scale: number;
  halo: boolean;
  targetFps: number;
  adaptive: boolean;
};

const MAX_PARTICLES = 16384;
const MAX_ITERATIONS = 1024;
const WORKGROUP_SIZE = 64;

const computeShader = /* wgsl */ `
struct Params {
  c: vec2f,
  aspect: f32,
  zoom: f32,
  particleCount: u32,
  iterations: u32,
  time: f32,
  spread: f32,
  center: vec2f,
  pad: vec2f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> vertices: array<vec2f>;

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

  let h1 = hash(particle * 2u + u32(params.time * 977.0));
  let h2 = hash(particle * 2u + 1u + u32(params.time * 577.0));
  let radius = sqrt(h1) * params.spread;
  let angle = h2 * 6.283185307;
  let sampleC = params.c + vec2f(cos(angle), sin(angle)) * radius;
  var z = vec2f(0.0);

  for (var step = 0u; step < params.iterations; step++) {
    let slot = particle * params.iterations + step;
    if (dot(z, z) > 256.0) {
      vertices[slot] = vec2f(4.0, 4.0);
      continue;
    }

    z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + sampleC;
    vertices[slot] = toClip(z);
  }
}
`;

const orbitShader = /* wgsl */ `
struct OrbitUniforms {
  alpha: f32,
  hue: f32,
  iterations: f32,
  pointSize: f32,
  resolution: vec2f,
  pad: vec2f,
}
@group(0) @binding(0) var<uniform> style: OrbitUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
}

@vertex
fn vs(@location(0) position: vec2f, @builtin(vertex_index) vertexIndex: u32) -> VSOut {
  var out: VSOut;
  out.position = vec4f(position, 0.0, 1.0);
  let t = f32(vertexIndex % u32(style.iterations)) / style.iterations;
  out.color = mix(vec3f(0.20, 1.0, 0.60), vec3f(0.30, 0.48, 1.0), t + style.hue);
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
  pad: vec2f,
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
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VSOut {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let clipOffset = corner * style.pointSize / style.resolution;
  var out: VSOut;
  out.position = vec4f(center + clipOffset, 0.0, 1.0);
  let t = f32(instanceIndex % u32(style.iterations)) / style.iterations;
  out.color = mix(vec3f(0.20, 1.0, 0.60), vec3f(0.30, 0.48, 1.0), t + style.hue);
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
  for (let i = 0; i < max; i++) {
    const nextR = zr * zr - zi * zi + cr;
    zi = 2 * zr * zi + ci;
    zr = nextR;
    if (zr * zr + zi * zi > 4) return `Escapes after ${i + 1} iterations`;
  }
  return `Bounded through ${max} iterations`;
}

export default function OrbitLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<{ update: (next: Partial<EngineSettings>) => void; reset: () => void } | null>(null);
  const pointRef = useRef({ x: -0.74364, y: 0.13183 });
  const settingsRef = useRef<EngineSettings>({
    iterations: 512,
    density: 13,
    persistence: 94,
    pointSize: 0.65,
    scale: 1,
    halo: true,
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
        size: MAX_PARTICLES * MAX_ITERATIONS * 8,
        usage: usage.STORAGE | usage.VERTEX,
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
          buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }],
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
            arrayStride: 8,
            stepMode: "instance",
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
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
      let textureSize = { width: 0, height: 0 };
      let pixelRatio = 1;
      let textureIndex = 0;
      let currentParticles = Math.min(MAX_PARTICLES, 2 ** settingsRef.current.density);
      let viewCenter = { x: -0.62, y: 0 };
      let zoom = 1;
      let dragging = false;
      let lastPointer = { x: 0, y: 0 };
      let frameCount = 0;
      let lastStatTime = performance.now();
      let lastFrameTime = performance.now();
      let smoothFrameMs = 16.7;

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
        const effectiveZoom = zoom * settingsRef.current.scale;
        return {
          x: viewCenter.x + nx * (2.35 * aspect) / effectiveZoom,
          y: viewCenter.y + ny * 2.35 / effectiveZoom,
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
          const effectiveZoom = zoom * settingsRef.current.scale;
          viewCenter.x -= dx * 4.7 * (rect.width / rect.height) / effectiveZoom;
          viewCenter.y += dy * 4.7 / effectiveZoom;
          lastPointer = { x: event.clientX, y: event.clientY };
          clearAccumulation();
          return;
        }
        const next = toComplex(event.clientX, event.clientY);
        pointRef.current = next;
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

      engineRef.current = {
        update(next) {
          if (typeof next.density === "number") currentParticles = Math.min(MAX_PARTICLES, 2 ** next.density);
          if (next.persistence !== undefined || next.iterations !== undefined || next.pointSize !== undefined || next.scale !== undefined || next.halo !== undefined) clearAccumulation();
        },
        reset() {
          viewCenter = { x: -0.62, y: 0 };
          zoom = 1;
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
        if (cfg.adaptive && frameCount % 30 === 0) {
          const targetMs = 1000 / cfg.targetFps;
          if (smoothFrameMs > targetMs * 1.12) currentParticles = Math.max(512, Math.floor(currentParticles * 0.82 / 64) * 64);
          else if (smoothFrameMs < targetMs * 0.82) currentParticles = Math.min(MAX_PARTICLES, 2 ** cfg.density, Math.ceil(currentParticles * 1.12 / 64) * 64);
        }

        const aspect = textureSize.width / textureSize.height;
        const point = pointRef.current;
        const effectiveZoom = zoom * cfg.scale;
        const params = new Float32Array(16);
        params[0] = point.x;
        params[1] = point.y;
        params[2] = aspect;
        params[3] = effectiveZoom;
        new Uint32Array(params.buffer)[4] = currentParticles;
        new Uint32Array(params.buffer)[5] = cfg.iterations;
        params[6] = now * 0.001;
        params[7] = 0.014 / Math.sqrt(effectiveZoom);
        params[8] = viewCenter.x;
        params[9] = viewCenter.y;
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
          0,
          0,
        ]));
        const fade = 0.84 + cfg.persistence * 0.00165;
        const gain = 1.5;
        device.queue.writeBuffer(fadeBuffer, 0, new Float32Array([fade, gain, cfg.halo ? 1 : 0, 0]));

        const source = textures[textureIndex];
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
        fadePass.setBindGroup(0, makeTextureBindGroup(fadePipeline, source));
        fadePass.draw(3);
        fadePass.end();

        const orbitPass = encoder.beginRenderPass({
          colorAttachments: [{ view: destination.createView(), loadOp: "load", storeOp: "store" }],
        });
        orbitPass.setVertexBuffer(0, vertexBuffer);
        const pointCount = currentParticles * cfg.iterations;
        if (cfg.pointSize <= 1) {
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
        displayPass.setBindGroup(0, makeTextureBindGroup(displayPipeline, destination));
        displayPass.draw(3);
        displayPass.end();
        device.queue.submit([encoder.finish()]);
        textureIndex = 1 - textureIndex;

        if (now - lastStatTime > 500) {
          const fps = 1000 / smoothFrameMs;
          const samples = currentParticles * cfg.iterations;
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
              <span className="controlValue">{settings.iterations} iterations</span>
            </span>
            <input className="range" type="range" min="32" max={MAX_ITERATIONS} step="32" value={settings.iterations} onChange={(event) => patchSettings({ iterations: Number(event.target.value) })} />
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
                <span className="controlLabel">View scale</span>
                <span className="controlValue">{settings.scale.toFixed(2)}×</span>
              </span>
              <input className="range" type="range" min="0.5" max="8" step="0.05" value={settings.scale} onChange={(event) => patchSettings({ scale: Number(event.target.value) })} />
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

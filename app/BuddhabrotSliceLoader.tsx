"use client";

/* WebGPU types are not part of this project's TypeScript DOM library yet. */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { useEffect, useRef, useState } from "react";
import styles from "./OfflineBuddhabrot.module.css";

const SEED_COUNT = 2_048;
const ITERATIONS = 160;
const WORKGROUP_SIZE = 64;
const POINT_STRIDE = 32;

const computeShader = /* wgsl */ `
struct Params {
  seedCount: u32,
  iterations: u32,
  epoch: u32,
  minOrbit: u32,
}

struct Point {
  center: vec4f,
  tangent: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> points: array<Point>;

fn hash(value: u32) -> f32 {
  var x = value;
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
  let random = vec2f(hash(randomBase), hash(randomBase ^ 0x9e3779b9u));
  let c = vec2f(mix(-2.12, 0.72, random.x), mix(-1.42, 1.42, random.y));
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
      points[slot].center = vec4f(0.0);
      points[slot].tangent = vec4f(0.0);
      continue;
    }

    let next = iterate(z, c);
    let after = iterate(next, c);
    let layer = log2(max(f32(escapeAt), 1.0)) / log2(f32(params.iterations));
    points[slot].center = vec4f(next, layer, 1.0);
    points[slot].tangent = vec4f(after - next, 0.0, layer);
    z = next;
  }
}
`;

const pointShader = /* wgsl */ `
struct Style {
  resolution: vec2f,
  slice: f32,
  thickness: f32,
  pointScale: f32,
  aspect: f32,
  time: f32,
  intensity: f32,
}

@group(0) @binding(0) var<uniform> style: Style;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) weight: f32,
}

@vertex
fn vs(
  @location(0) centerData: vec4f,
  @location(1) tangentData: vec4f,
  @builtin(vertex_index) vertexIndex: u32,
) -> VertexOut {
  let corners = array<vec2f, 6>(
    vec2f(-3.0, -3.0), vec2f(3.0, -3.0), vec2f(-3.0, 3.0),
    vec2f(-3.0, 3.0), vec2f(3.0, -3.0), vec2f(3.0, 3.0)
  );
  var out: VertexOut;
  let layerDelta = abs(centerData.z - style.slice);
  let visible = centerData.w > 0.5 && layerDelta < style.thickness * 3.2;
  if (!visible) {
    out.position = vec4f(2.0, 2.0, 0.0, 1.0);
    out.local = vec2f(99.0);
    out.weight = 0.0;
    return out;
  }

  let rawCenter = centerData.xy - vec2f(-0.50, 0.0);
  var center = vec2f(-rawCenter.x, rawCenter.y) * 0.46;
  center.x /= style.aspect;
  var tangent = vec2f(-tangentData.x, tangentData.y);
  tangent.x /= style.aspect;
  let tangentLength = max(length(tangent), 0.0001);
  let majorDirection = tangent / tangentLength;
  let minorDirection = vec2f(-majorDirection.y, majorDirection.x);
  let shimmer = 0.94 + 0.06 * sin(style.time * 1.7 + centerData.z * 17.0);
  let sigma = style.pointScale * shimmer;
  let corner = corners[vertexIndex];
  let pixelOffset = majorDirection * corner.x * sigma * 1.18 + minorDirection * corner.y * sigma * 0.88;
  let ndcOffset = pixelOffset * 2.0 / style.resolution;
  let normalizedDelta = layerDelta / max(style.thickness, 0.0001);

  out.position = vec4f(center + ndcOffset, 0.0, 1.0);
  out.local = corner;
  out.weight = exp(-0.5 * normalizedDelta * normalizedDelta) * style.intensity;
  return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let radiusSquared = dot(in.local, in.local);
  if (radiusSquared > 9.0) { discard; }
  let gaussian = exp(-0.5 * radiusSquared);
  let light = gaussian * in.weight;
  return vec4f(light, light, light, light);
}
`;

const displayShader = /* wgsl */ `
struct Display {
  resolution: vec2f,
  exposure: f32,
  time: f32,
}

@group(0) @binding(0) var lightTexture: texture_2d<f32>;
@group(0) @binding(1) var lightSampler: sampler;
@group(0) @binding(2) var<uniform> display: Display;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  let positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VertexOut;
  out.position = vec4f(positions[index], 0.0, 1.0);
  out.uv = positions[index] * vec2f(0.5, -0.5) + 0.5;
  return out;
}

fn sampleLight(uv: vec2f) -> f32 {
  return textureSample(lightTexture, lightSampler, uv).r;
}

@fragment
fn fadeFs(in: VertexOut) -> @location(0) vec4f {
  return textureSample(lightTexture, lightSampler, in.uv) * 0.89;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let texel = 1.0 / display.resolution;
  let center = sampleLight(in.uv);
  let nearGlow = (
    sampleLight(in.uv + vec2f(texel.x * 2.0, 0.0)) +
    sampleLight(in.uv - vec2f(texel.x * 2.0, 0.0)) +
    sampleLight(in.uv + vec2f(0.0, texel.y * 2.0)) +
    sampleLight(in.uv - vec2f(0.0, texel.y * 2.0))
  ) * 0.25;
  let farGlow = (
    sampleLight(in.uv + vec2f(texel.x * 7.0, texel.y * 4.0)) +
    sampleLight(in.uv + vec2f(-texel.x * 7.0, texel.y * 4.0)) +
    sampleLight(in.uv + vec2f(texel.x * 7.0, -texel.y * 4.0)) +
    sampleLight(in.uv - vec2f(texel.x * 7.0, texel.y * 4.0))
  ) * 0.25;
  let density = log(1.0 + center * display.exposure);
  let glow = log(1.0 + (nearGlow * 0.75 + farGlow * 0.4) * display.exposure);

  let electricBlue = vec3f(0.015, 0.10, 0.92);
  let warmGold = vec3f(1.0, 0.47, 0.065);
  let hotWhite = vec3f(1.0, 0.96, 0.76);
  var color = electricBlue * glow * 1.25;
  color += mix(electricBlue, warmGold, smoothstep(0.12, 0.62, density)) * density;
  color = mix(color, hotWhite * (0.68 + density), smoothstep(0.82, 1.48, density));

  let centered = in.uv * 2.0 - 1.0;
  let vignette = 1.0 - smoothstep(0.45, 1.35, dot(centered, centered));
  let scan = 0.985 + 0.015 * sin(in.uv.y * display.resolution.y * 0.36 + display.time);
  color *= vignette * scan;
  color = color / (vec3f(1.0) + color);
  color = pow(color, vec3f(0.82));
  return vec4f(vec3f(0.0015, 0.0025, 0.008) + color, 1.0);
}
`;

type Props = {
  exiting: boolean;
};

export default function BuddhabrotSliceLoader({ exiting }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [gpuReady, setGpuReady] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gpu = (navigator as Navigator & { gpu?: any }).gpu;
    if (!canvas || !gpu) return;
    const targetCanvas: HTMLCanvasElement = canvas;

    let disposed = false;
    let frame = 0;
    let cleanup: (() => void) | undefined;

    async function start() {
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter || disposed) return;
      const device = await adapter.requestDevice();
      if (disposed) {
        device.destroy();
        return;
      }

      const context = targetCanvas.getContext("webgpu") as any;
      if (!context) {
        device.destroy();
        return;
      }
      const format = gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });

      const bufferUsage = (globalThis as any).GPUBufferUsage;
      const textureUsage = (globalThis as any).GPUTextureUsage;
      const pointBuffer = device.createBuffer({
        size: SEED_COUNT * ITERATIONS * POINT_STRIDE,
        usage: bufferUsage.STORAGE | bufferUsage.VERTEX,
      });
      const paramsBuffer = device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
      const styleBuffer = device.createBuffer({ size: 32, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
      const displayBuffer = device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
      const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

      const computeModule = device.createShaderModule({ code: computeShader });
      const pointModule = device.createShaderModule({ code: pointShader });
      const displayModule = device.createShaderModule({ code: displayShader });
      const [computePipeline, pointPipeline, fadePipeline, displayPipeline] = await Promise.all([
        device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: computeModule, entryPoint: "main" },
        }),
        device.createRenderPipelineAsync({
          layout: "auto",
          vertex: {
            module: pointModule,
            entryPoint: "vs",
            buffers: [{
              arrayStride: POINT_STRIDE,
              stepMode: "instance",
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x4" },
                { shaderLocation: 1, offset: 16, format: "float32x4" },
              ],
            }],
          },
          fragment: {
            module: pointModule,
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
        }),
        device.createRenderPipelineAsync({
          layout: "auto",
          vertex: { module: displayModule, entryPoint: "vs" },
          fragment: { module: displayModule, entryPoint: "fadeFs", targets: [{ format: "rgba16float" }] },
          primitive: { topology: "triangle-list" },
        }),
        device.createRenderPipelineAsync({
          layout: "auto",
          vertex: { module: displayModule, entryPoint: "vs" },
          fragment: { module: displayModule, entryPoint: "fs", targets: [{ format }] },
          primitive: { topology: "triangle-list" },
        }),
      ]);
      if (disposed) {
        pointBuffer.destroy();
        paramsBuffer.destroy();
        styleBuffer.destroy();
        displayBuffer.destroy();
        device.destroy();
        return;
      }

      const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: pointBuffer } },
        ],
      });
      const pointBindGroup = device.createBindGroup({
        layout: pointPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: styleBuffer } }],
      });

      let lightTextures: any[] = [];
      let fadeBindGroups: any[] = [];
      let displayBindGroups: any[] = [];
      let textureSize = { width: 0, height: 0 };
      let textureIndex = 0;
      let epoch = 7;
      let drawSeeds = SEED_COUNT;
      let smoothFrameMs = 16.7;
      let previousTime = performance.now();
      let qualityTimer = 0;
      let reportedReady = false;
      let renderedFrames = 0;
      let running = !document.hidden;
      const startedAt = performance.now();
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      function resize() {
        const rect = targetCanvas.getBoundingClientRect();
        const dprLimit = rect.width < 720 ? 1 : 1.25;
        const dpr = Math.min(window.devicePixelRatio || 1, dprLimit);
        const width = Math.max(1, Math.round(rect.width * dpr));
        const height = Math.max(1, Math.round(rect.height * dpr));
        if (width === textureSize.width && height === textureSize.height) return;
        targetCanvas.width = width;
        targetCanvas.height = height;
        lightTextures.forEach((texture) => texture.destroy());
        lightTextures = [0, 1].map(() => device.createTexture({
          size: [width, height],
          format: "rgba16float",
          usage: textureUsage.RENDER_ATTACHMENT | textureUsage.TEXTURE_BINDING,
        }));
        const makeTextureBindGroup = (pipeline: any, texture: any, includeDisplay: boolean) => device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: sampler },
            ...(includeDisplay ? [{ binding: 2, resource: { buffer: displayBuffer } }] : []),
          ],
        });
        fadeBindGroups = lightTextures.map((texture) => makeTextureBindGroup(fadePipeline, texture, false));
        displayBindGroups = lightTextures.map((texture) => makeTextureBindGroup(displayPipeline, texture, true));
        textureSize = { width, height };
        textureIndex = 0;
        const clearEncoder = device.createCommandEncoder();
        for (const texture of lightTextures) {
          const clearPass = clearEncoder.beginRenderPass({
            colorAttachments: [{
              view: texture.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
            }],
          });
          clearPass.end();
        }
        device.queue.submit([clearEncoder.finish()]);
      }

      const resizeObserver = new ResizeObserver(resize);
      resizeObserver.observe(targetCanvas);
      resize();

      function draw(now: number) {
        frame = 0;
        if (disposed || !running || lightTextures.length !== 2 || displayBindGroups.length !== 2) return;
        const frameMs = Math.min(100, now - previousTime);
        previousTime = now;
        smoothFrameMs = smoothFrameMs * 0.92 + frameMs * 0.08;
        qualityTimer += frameMs;
        if (qualityTimer > 600) {
          if (smoothFrameMs > 25 && drawSeeds > 768) drawSeeds = Math.max(768, drawSeeds - 256);
          else if (smoothFrameMs < 18 && drawSeeds < SEED_COUNT) drawSeeds = Math.min(SEED_COUNT, drawSeeds + 256);
          qualityTimer = 0;
        }

        const seconds = (now - startedAt) * 0.001;
        const phase = reduceMotion ? 0.38 : (seconds % 6.9) / 6.9;
        const slice = 0.18 + 0.78 * (0.5 - 0.5 * Math.cos(phase * Math.PI * 2));
        const pointScale = Math.max(0.48, Math.min(1.2, textureSize.height / 930));
        device.queue.writeBuffer(styleBuffer, 0, new Float32Array([
          textureSize.width,
          textureSize.height,
          slice,
          0.075,
          pointScale,
          textureSize.width / textureSize.height,
          seconds,
          0.040,
        ]));
        device.queue.writeBuffer(displayBuffer, 0, new Float32Array([
          textureSize.width,
          textureSize.height,
          15.0,
          seconds * 2.0,
        ]));
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([drawSeeds, ITERATIONS, epoch++, 7]));

        const encoder = device.createCommandEncoder();
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(drawSeeds / WORKGROUP_SIZE));
        computePass.end();

        const destinationIndex = 1 - textureIndex;
        const fadePass = encoder.beginRenderPass({
          colorAttachments: [{
            view: lightTextures[destinationIndex].createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          }],
        });
        fadePass.setPipeline(fadePipeline);
        fadePass.setBindGroup(0, fadeBindGroups[textureIndex]);
        fadePass.draw(3);
        fadePass.end();

        const pointPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: lightTextures[destinationIndex].createView(),
            loadOp: "load",
            storeOp: "store",
          }],
        });
        pointPass.setPipeline(pointPipeline);
        pointPass.setBindGroup(0, pointBindGroup);
        pointPass.setVertexBuffer(0, pointBuffer);
        pointPass.draw(6, drawSeeds * ITERATIONS);
        pointPass.end();

        const displayPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        });
        displayPass.setPipeline(displayPipeline);
        displayPass.setBindGroup(0, displayBindGroups[destinationIndex]);
        displayPass.draw(3);
        displayPass.end();
        device.queue.submit([encoder.finish()]);
        textureIndex = destinationIndex;
        renderedFrames++;

        if (!reportedReady) {
          reportedReady = true;
          setGpuReady(true);
        }
        if (!reduceMotion || renderedFrames < 16) frame = requestAnimationFrame(draw);
      }

      const onVisibilityChange = () => {
        running = !document.hidden;
        if (!running) {
          cancelAnimationFrame(frame);
          frame = 0;
        } else if (!frame && !disposed) {
          previousTime = performance.now();
          frame = requestAnimationFrame(draw);
        }
      };
      document.addEventListener("visibilitychange", onVisibilityChange);
      if (running) frame = requestAnimationFrame(draw);
      device.lost.then(() => {
        if (!disposed) setGpuReady(false);
      });

      cleanup = () => {
        resizeObserver.disconnect();
        document.removeEventListener("visibilitychange", onVisibilityChange);
        cancelAnimationFrame(frame);
        lightTextures.forEach((texture) => texture.destroy());
        pointBuffer.destroy();
        paramsBuffer.destroy();
        styleBuffer.destroy();
        displayBuffer.destroy();
        context.unconfigure();
        device.destroy();
      };
    }

    start().catch(() => undefined);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      cleanup?.();
    };
  }, []);

  return (
    <div
      className={`${styles.sliceLoader} ${exiting ? styles.sliceLoaderExit : ""}`}
      role="status"
      aria-live="polite"
    >
      <div className={styles.sliceFallback} aria-hidden="true" />
      <canvas
        ref={canvasRef}
        className={`${styles.sliceCanvas} ${gpuReady ? styles.sliceCanvasReady : ""}`}
        aria-label="GPU-computed Buddhabrot depth slices"
      />
      <div className={styles.sliceReadout}>
        <span className={styles.sliceIndicator} aria-hidden="true" />
        Computing orbit slice · loading 1M splats
      </div>
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import styles from "./OfflineBuddhabrot.module.css";

type AssetStats = {
  candidateSamples: number;
  mirroredSamples: number;
  escapedSamples: number;
  maxIterations: number;
  resolution: [number, number, number];
  gaussians: number;
  volumeAxis: string;
  mapPower: number;
  volumeDepth: number;
};

const FALLBACK_STATS: AssetStats = {
  candidateSamples: 12_000_000,
  mirroredSamples: 24_000_000,
  escapedSamples: 326_450,
  maxIterations: 1_048_576,
  resolution: [1600, 1600, 256],
  gaussians: 1_000_000,
  volumeAxis: "normalized-orbit-progress",
  mapPower: 2,
  volumeDepth: 1.15,
};

function compact(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return value.toLocaleString();
}

export default function OfflineBuddhabrot() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<AssetStats>(FALLBACK_STATS);

  useEffect(() => {
    fetch("/buddhabrot.json")
      .then((response) => response.json() as Promise<AssetStats>)
      .then(setStats)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x030408);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
    const view = {
      target: new THREE.Vector3(-0.25, 0, 0),
      yaw: 0,
      pitch: 0,
      distance: 5.1,
    };
    const panRight = new THREE.Vector3();
    const panUp = new THREE.Vector3();
    const forward = new THREE.Vector3();
    let scheduledFrame = 0;

    function render() {
      const cosPitch = Math.cos(view.pitch);
      camera.position.set(
        view.target.x + Math.sin(view.yaw) * cosPitch * view.distance,
        view.target.y + Math.sin(view.pitch) * view.distance,
        view.target.z + Math.cos(view.yaw) * cosPitch * view.distance,
      );
      camera.lookAt(view.target);
      renderer.render(scene, camera);
    }

    function scheduleRender() {
      if (scheduledFrame || disposed) return;
      scheduledFrame = requestAnimationFrame(() => {
        scheduledFrame = 0;
        render();
      });
    }

    const spark = new SparkRenderer({
      renderer,
      onDirty: scheduleRender,
      minPixelRadius: 0.02,
      maxStdDev: 2.15,
      blurAmount: 0,
      falloff: 1,
      sortRadial: true,
    });
    scene.add(spark);

    const splat = new SplatMesh({ url: "/buddhabrot.spz", lod: false });
    splat.opacity = 0.85;
    // Canonical Buddhabrot presentation: the real axis is vertical.
    splat.rotation.z = Math.PI / 2;
    splat.position.y = 0.5;
    scene.add(splat);

    let dragging = false;
    let panning = false;
    let lastPointer = { x: 0, y: 0 };

    function resetCamera() {
      view.target.set(-0.25, 0, 0);
      view.yaw = 0;
      view.pitch = 0;
      view.distance = 5.1;
      scheduleRender();
    }

    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0 && event.button !== 2) return;
      dragging = true;
      panning = event.shiftKey || event.button === 2;
      lastPointer = { x: event.clientX, y: event.clientY };
      renderer.domElement.setPointerCapture(event.pointerId);
    };
    const onPointerUp = () => { dragging = false; };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      if (panning || event.shiftKey) {
        forward.copy(view.target).sub(camera.position).normalize();
        panRight.crossVectors(forward, camera.up).normalize();
        panUp.crossVectors(panRight, forward).normalize();
        const scale = view.distance * 0.00135;
        view.target.addScaledVector(panRight, -dx * scale);
        view.target.addScaledVector(panUp, dy * scale);
      } else {
        view.yaw -= dx * 0.006;
        view.pitch = THREE.MathUtils.clamp(view.pitch + dy * 0.005, -1.35, 1.35);
      }
      lastPointer = { x: event.clientX, y: event.clientY };
      scheduleRender();
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      view.distance = THREE.MathUtils.clamp(view.distance * Math.exp(event.deltaY * 0.001), 2.1, 9);
      scheduleRender();
    };
    const onDoubleClick = () => resetCamera();
    const onContextMenu = (event: MouseEvent) => event.preventDefault();

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("pointercancel", onPointerUp);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
    renderer.domElement.addEventListener("dblclick", onDoubleClick);
    renderer.domElement.addEventListener("contextmenu", onContextMenu);

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      scheduleRender();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    splat.initialized
      .then(() => {
        if (!disposed) {
          setLoaded(true);
          scheduleRender();
        }
      })
      .catch((reason: unknown) => {
        if (!disposed) setError(reason instanceof Error ? reason.message : "Could not load the splat asset.");
      });

    return () => {
      disposed = true;
      observer.disconnect();
      cancelAnimationFrame(scheduledFrame);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerUp);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.domElement.removeEventListener("dblclick", onDoubleClick);
      renderer.domElement.removeEventListener("contextmenu", onContextMenu);
      splat.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.mark} aria-hidden="true" />
          <span>
            <strong>3D Buddhabrot</strong>
            <small>ORBIT-TIME ESCAPE VOLUME</small>
          </span>
        </div>
        <p>One finished 3D escape field. No browser-side fractal iteration.</p>
        <a href="/orbit">Open orbit lab →</a>
      </header>

      <section className={styles.viewport} aria-label="Precomputed 3D Buddhabrot Gaussian splat viewer">
        <div ref={hostRef} className={styles.canvasHost} aria-label="Drag to rotate. Shift-drag to pan. Scroll to zoom. Double-click to reset." />
        <div className={styles.vignette} aria-hidden="true" />

        <aside className={styles.card}>
          <span className={styles.eyebrow}><i /> Precomputed XYZ artifact</span>
          <h1>THE ESCAPE VOLUME</h1>
          <p>
            The canonical Buddhabrot in X/Y, sculpted into continuous depth by
            each point&apos;s progress along its escaping orbit. Front view keeps
            the Buddha; rotation opens the trajectories inside it.
          </p>
          <dl>
            <div><dt>Quadratic power</dt><dd>{stats.mapPower}</dd></div>
            <div><dt>Escape cap</dt><dd>{compact(stats.maxIterations)}</dd></div>
            <div><dt>Candidate c values</dt><dd>{compact(stats.candidateSamples)}</dd></div>
            <div><dt>Effective samples</dt><dd>{compact(stats.mirroredSamples)}</dd></div>
            <div><dt>Escaping paths</dt><dd>{compact(stats.escapedSamples)}</dd></div>
            <div><dt>Tiny Gaussians</dt><dd>{compact(stats.gaussians)}</dd></div>
            <div><dt>Orbit-time lattice</dt><dd>{stats.resolution[0]}² × {stats.resolution[2]}</dd></div>
          </dl>
        </aside>

        <p className={styles.hint}>DRAG TO ROTATE · SHIFT-DRAG TO PAN · SCROLL TO ZOOM · DOUBLE-CLICK TO RESET</p>
        {!loaded && !error && <div className={styles.loading}><span /> Loading 1M transparent XYZ splats</div>}
        {error && <div className={`${styles.loading} ${styles.error}`}>Could not load splat: {error}</div>}
      </section>
    </main>
  );
}

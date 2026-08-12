import type { Metadata } from "next";
import OrbitLab from "../OrbitLab";
import "./orbit-layout.css";

export const metadata: Metadata = {
  title: "Orbit Lab — Progressive Mandelbrot Iterator",
  description:
    "A cursor-driven WebGPU Mandelbrot iterator that resumes deep orbits across animation frames.",
};

export default function OrbitPage() {
  return <OrbitLab />;
}

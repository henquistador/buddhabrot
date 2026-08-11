import type { Metadata } from "next";
import OrbitLab from "./OrbitLab";

export const metadata: Metadata = {
  title: "Orbit Lab",
  description: "A GPU-powered Mandelbrot orbit explorer and live Buddhabrot precursor.",
};

export default function Home() {
  return <OrbitLab />;
}

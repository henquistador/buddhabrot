import type { Metadata } from "next";
import OfflineBuddhabrot from "./OfflineBuddhabrot";

export const metadata: Metadata = {
  title: "3D Buddhabrot — Mandelbulb Escape Volume",
  description: "A genuine XYZ Buddhabrot made from power-8 Mandelbulb escape orbits and 650,000 tiny Gaussian splats.",
};

export default function Home() {
  return <OfflineBuddhabrot />;
}

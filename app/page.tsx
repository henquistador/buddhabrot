import type { Metadata } from "next";
import OfflineBuddhabrot from "./OfflineBuddhabrot";

export const metadata: Metadata = {
  title: "3D Buddhabrot — Mandelbulb Escape Volume",
  description: "A genuine XYZ Buddhabrot at 864³ resolution, made from power-8 Mandelbulb escape orbits and one million transparent Gaussian splats.",
};

export default function Home() {
  return <OfflineBuddhabrot />;
}

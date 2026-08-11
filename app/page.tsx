import type { Metadata } from "next";
import BuddhabrotSplat from "./BuddhabrotSplat";

export const metadata: Metadata = {
  title: "Buddhabrot Splat Lab",
  description: "A live 3D Gaussian-splat Buddhabrot renderer powered by WebGPU.",
};

export default function Home() {
  return <BuddhabrotSplat />;
}

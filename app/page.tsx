import type { Metadata } from "next";
import OfflineBuddhabrot from "./OfflineBuddhabrot";

export const metadata: Metadata = {
  title: "Buddhabrot — Offline Gaussian Exposure",
  description: "A one-million-iteration Buddhabrot baked into a rotatable 3D volume of 600,000 tiny Gaussian splats.",
};

export default function Home() {
  return <OfflineBuddhabrot />;
}

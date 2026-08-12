import type { Metadata } from "next";
import OfflineBuddhabrot from "./OfflineBuddhabrot";

export const metadata: Metadata = {
  title: "3D Buddhabrot — Orbit-Time Escape Volume",
  description: "A proper offline Buddhabrot at 1600² × 256 resolution, with continuous orbit progress forming its explorable third dimension.",
};

export default function Home() {
  return <OfflineBuddhabrot />;
}

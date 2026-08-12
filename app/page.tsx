import type { Metadata } from "next";
import OfflineBuddhabrot from "./OfflineBuddhabrot";

export const metadata: Metadata = {
  title: "3D Buddhabrot — Quaternion Escape Volume",
  description: "A genuine XYZ Buddhabrot at 864³ resolution, preserving the classic Buddha cross-section and its faint interior escape trails.",
};

export default function Home() {
  return <OfflineBuddhabrot />;
}

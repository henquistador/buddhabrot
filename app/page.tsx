import type { Metadata } from "next";
import OfflineBuddhabrot from "./OfflineBuddhabrot";

export const metadata: Metadata = {
  title: "3D Buddhabrot — Complex Hénon Escape Cloud",
  description: "A million-splat cloud formed by projecting coupled quadratic escape orbits from two complex variables into real XYZ space.",
};

export default function Home() {
  return <OfflineBuddhabrot />;
}

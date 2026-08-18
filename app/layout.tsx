import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const siteUrl = process.env.SITE_URL ?? "http://localhost:3000";
const imageUrl = `${basePath}/og.png`;
const title = "3D Buddhabrot — Complex Hénon Escape Cloud";
const description = "A million-splat cloud formed by projecting coupled quadratic escape orbits from two complex variables into real XYZ space.";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title,
  description,
  icons: { icon: `${basePath}/favicon.svg` },
  openGraph: {
    title,
    description,
    type: "website",
    images: [{ url: imageUrl, width: 1536, height: 1024, alt: "Buddhabrot 3D Escape Volume" }],
  },
  twitter: { card: "summary_large_image", title, description, images: [imageUrl] },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}

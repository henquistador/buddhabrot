import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the offline Buddhabrot artifact viewer", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>3D Buddhabrot — Complex Hénon Escape Cloud/);
  assert.match(html, /One finished 3D escape field/i);
  assert.match(html, /Precomputed 3D Buddhabrot Gaussian splat viewer/);
  assert.match(html, /Quadratic power/);
  assert.match(html, /Tiny Gaussians/);
  assert.match(html, /XYZ lattice/);
  assert.match(html, /Drag to rotate/i);
  assert.match(html, /Open orbit lab/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

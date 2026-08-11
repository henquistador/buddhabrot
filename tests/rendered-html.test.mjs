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

test("server-renders Buddhabrot Splat Lab without the disposable starter", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Buddhabrot Splat Lab/);
  assert.match(html, /Escape-orbit volume/i);
  assert.match(html, /Interactive 3D Buddhabrot Gaussian splat/);
  assert.match(html, /Auto · 60 FPS/);
  assert.match(html, /Splat scale/);
  assert.match(html, /Tangent stretch/);
  assert.match(html, /Bloom/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

// Feasibility check: can tfjs run on WebGPU (via Dawn bindings) in this runtime?
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
Object.defineProperty(globalThis.navigator, "gpu", { value: create([]), configurable: true });

const tf = await import("@tensorflow/tfjs-core");
await import("@tensorflow/tfjs-backend-cpu");
await import("@tensorflow/tfjs-backend-webgpu");

const adapter = await navigator.gpu.requestAdapter();
console.log(
  "adapter:",
  adapter?.info?.vendor,
  adapter?.info?.architecture,
  adapter?.info?.description,
);

const N = Number(process.argv[2] ?? 2048);

for (const name of ["cpu", "webgpu"]) {
  await tf.setBackend(name);
  await tf.ready();
  const a = tf.randomNormal([N, N]);
  const b = tf.randomNormal([N, N]);

  // warm-up (shader compilation, buffer allocation)
  await tf.matMul(a, b).data();

  const t0 = performance.now();
  const c = tf.matMul(a, b);
  const data = await c.data();
  const ms = performance.now() - t0;

  console.log(
    `${tf.getBackend()}: matmul ${N}x${N} in ${ms.toFixed(1)} ms (c[0]=${data[0]?.toFixed(3)})`,
  );
  tf.dispose([a, b, c]);
}

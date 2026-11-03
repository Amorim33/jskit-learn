// Compares predictBatch on cpu vs webgpu for a large batch.
// Usage: bun scripts/bench.ts [samples] [features]
import { Perceptron, setBackend } from "../src";

const samples = Number(process.argv[2] ?? 100_000);
const features = Number(process.argv[3] ?? 100);

// linearly separable data: label = 1 when the sum of the features is positive
const X = Array.from({ length: samples }, () =>
  Array.from({ length: features }, () => Math.random() * 2 - 1),
);
const y = X.map((x) => (x.reduce((s, v) => s + v, 0) > 0 ? 1 : 0));

console.log(`${samples} samples x ${features} features`);

for (const name of ["cpu", "webgpu"] as const) {
  const active = await setBackend(name);
  const p = new Perceptron({ inputSize: features, learningRate: 0.01 });

  // warm-up (shader compilation, buffer allocation)
  await p.predictBatch(X.slice(0, 10));

  let t0 = performance.now();
  await p.train(X, y, 20);
  const trainMs = performance.now() - t0;

  t0 = performance.now();
  const predictions = await p.predictBatch(X);
  const predictMs = performance.now() - t0;

  const accuracy = predictions.filter((v, i) => v === y[i]).length / samples;
  console.log(
    `${active.padEnd(6)} train(20 epochs) ${trainMs.toFixed(0).padStart(6)} ms | ` +
      `predictBatch ${predictMs.toFixed(0).padStart(5)} ms | accuracy ${(accuracy * 100).toFixed(1)}%`,
  );
}

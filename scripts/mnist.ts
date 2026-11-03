// Trains the perceptron on MNIST to answer "is this digit a 3?" and draws the model.
// Downloads MNIST once into data/mnist (gitignored).
// Usage: bun scripts/mnist.ts [epochs] [out.svg]
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { Perceptron } from "../src";

const MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist";
const DATA = join(import.meta.dir, "../data/mnist");
const TRAIN = 10_000;
const TEST = 2_000;
const DIGIT = 3;

const epochs = Number(process.argv[2] ?? 50);
const out = process.argv[3] ?? join(import.meta.dir, "../figures/mnist-perceptron.svg");

/** Downloads (once) and gunzips an MNIST IDX file. */
async function load(name: string): Promise<Uint8Array> {
  const path = join(DATA, name);
  const file = Bun.file(path);
  if (await file.exists()) return new Uint8Array(await file.arrayBuffer());

  console.log(`downloading ${name}…`);
  const res = await fetch(`${MIRROR}/${name}.gz`);
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  const bytes = Bun.gunzipSync(new Uint8Array(await res.arrayBuffer()));
  await Bun.write(path, bytes);
  return bytes;
}

/** Parses an IDX image file into rows of pixels scaled to [0, 1]. */
function images(buf: Uint8Array, limit: number): number[][] {
  const view = new DataView(buf.buffer, buf.byteOffset);
  if (view.getUint32(0) !== 0x803) throw new Error("not an IDX image file");
  const n = Math.min(view.getUint32(4), limit);
  const size = view.getUint32(8) * view.getUint32(12);
  return Array.from({ length: n }, (_, i) =>
    Array.from(buf.subarray(16 + i * size, 16 + (i + 1) * size), (v) => v / 255),
  );
}

/** Parses an IDX label file into binary targets: 1 when the digit is {@link DIGIT}. */
function labels(buf: Uint8Array, limit: number): number[] {
  const view = new DataView(buf.buffer, buf.byteOffset);
  if (view.getUint32(0) !== 0x801) throw new Error("not an IDX label file");
  const n = Math.min(view.getUint32(4), limit);
  return Array.from(buf.subarray(8, 8 + n), (d) => (d === DIGIT ? 1 : 0));
}

const accuracy = (pred: number[], y: number[]) =>
  pred.filter((p, i) => p === y[i]).length / y.length;

await mkdir(DATA, { recursive: true });
const [Xtr, ytr, Xte, yte] = await Promise.all([
  load("train-images-idx3-ubyte").then((b) => images(b, TRAIN)),
  load("train-labels-idx1-ubyte").then((b) => labels(b, TRAIN)),
  load("t10k-images-idx3-ubyte").then((b) => images(b, TEST)),
  load("t10k-labels-idx1-ubyte").then((b) => labels(b, TEST)),
]);

console.log(
  `${Xtr.length} train / ${Xte.length} test, ${Xtr[0]?.length} features, target: "is it a ${DIGIT}?"`,
);

const p = new Perceptron({ inputSize: Xtr[0]?.length ?? 784, learningRate: 0.01 });
const t0 = performance.now();
await p.train(Xtr, ytr, epochs);
console.log(`trained ${epochs} max epochs in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const [trainPred, testPred] = await Promise.all([p.predictBatch(Xtr), p.predictBatch(Xte)]);
console.log(`train accuracy ${(accuracy(trainPred, ytr) * 100).toFixed(2)}%`);
console.log(`test accuracy  ${(accuracy(testPred, yte) * 100).toFixed(2)}%`);

await mkdir(join(out, ".."), { recursive: true });
await p.draw(out, { title: `Perceptron on MNIST: “is it a ${DIGIT}?”` });
console.log(`figure → ${out}`);

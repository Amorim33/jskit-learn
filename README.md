# jskit-learn

Machine learning methods implemented from scratch in TypeScript.

The public API takes and returns plain JS arrays. Under the hood, operations run on
[TensorFlow.js](https://github.com/tensorflow/tfjs), on the CPU or on the GPU via WebGPU.

```ts
import { Perceptron, setBackend } from "jskit-learn";

await setBackend("webgpu"); // optional, defaults to "cpu"

const p = new Perceptron({ inputSize: 2 });
await p.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1], 100);
await p.predictBatch([[1, 1], [0, 1]]); // [1, 0]
```

Every model also has a didactic version in plain JavaScript (no tensors, synchronous),
written to be read alongside the theory:

```ts
import { PlainPerceptron } from "jskit-learn";

const p = new PlainPerceptron({ inputSize: 2 });
p.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1], 100);
p.predict([1, 1]); // 1
```

## Backends

| Backend  | Runs on                                                                 |
| -------- | ----------------------------------------------------------------------- |
| `cpu`    | Anywhere. Default; deterministic, used by the tests.                    |
| `webgpu` | Browser (`navigator.gpu`), or Bun/Node via the optional `webgpu` package (Dawn bindings): Metal on macOS, Vulkan on Linux, D3D12 on Windows. |

`setBackend` resolves to the backend actually in use: if WebGPU is unavailable it falls
back to `cpu` with a warning. Pick the backend before creating models.

The GPU only pays off for large inputs: with a handful of samples, transfer and dispatch
overhead make it slower than `cpu`.

## Figures

`Perceptron.draw` writes a paper-ready SVG of the model: inputs and bias feeding Σ through
the learned weights, the step activation, and booktabs-style tables with the hyperparameters
and the stats of the last `train` call (epochs run, samples).

```ts
await p.train(X, y, 100);
await p.draw("perceptron.svg", { title: "Perceptron learning AND" });
```

Use it in LaTeX with `\includesvg` (package `svg`), or convert it to PDF with Inkscape.

## Development

```sh
bun install
bun test          # run tests (GPU math tests are skipped when no GPU is found)
bun run typecheck # type-check src and tests
bun run lint      # lint + format check (Biome)
bun run build     # emit ESM + .d.ts to dist/
bun run bench     # cpu vs webgpu: bun scripts/bench.ts [samples] [features]
```

`bun scripts/webgpu-spike.ts [N]` checks that WebGPU works in the current runtime (N×N matmul).

## Architecture

- `src/linear_model/`: models, written only against `src/math`. They never see the backend.
- `src/linear_model/plain/` and `src/math/plain/`: the didactic plain-JS versions, on `number[]`.
- `src/math/`: the math abstractions: `Matrix` plus operations (`matmul`, `heavside`,
  `scalarProduct`, ...). They run on the active backend; a `Matrix`'s backend memory is
  released automatically when it is garbage-collected.
- `src/backend/`: internal `Backend` interface and its TensorFlow.js implementation.
  Swapping the engine (e.g. for hand-written kernels) only touches this folder.

## Roadmap

- [x] Perceptron (`src/linear_model/perceptron.ts`, plain: `src/linear_model/plain/perceptron.ts`)

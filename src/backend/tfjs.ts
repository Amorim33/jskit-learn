import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";

import type { Backend, BackendName, Tensor } from "./types";

// Silences tfjs warnings (e.g. the "install tfjs-node" banner) and skips debug-only checks.
tf.env().set("PROD", true);

const toTf = (t: Tensor) => t as unknown as tf.Tensor;
const fromTf = (t: tf.Tensor) => t as unknown as Tensor;

/**
 * Makes `navigator.gpu` available outside the browser, using Dawn's bindings.
 *
 * @returns Whether a WebGPU implementation is available.
 */
const installWebGPU = async () => {
  const nav = globalThis.navigator as { gpu?: unknown } | undefined;
  if (nav?.gpu) return true;

  try {
    const { create, globals } = await import("webgpu");
    Object.assign(globalThis, globals);
    if (!globalThis.navigator) {
      Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });
    }
    Object.defineProperty(globalThis.navigator, "gpu", { value: create([]), configurable: true });
    return true;
  } catch {
    return false;
  }
};

/**
 * Activates a TensorFlow.js backend, falling back to `cpu` when WebGPU is unavailable.
 *
 * @remarks
 * Without calling it, tfjs initializes `cpu` on the first operation.
 *
 * @param name - Backend to activate.
 * @returns The backend that is actually running.
 */
export const activateTfjsBackend = async (name: BackendName): Promise<BackendName> => {
  if (name === "webgpu") {
    if (await installWebGPU()) {
      await import("@tensorflow/tfjs-backend-webgpu");
      // setBackend resolves to false (it doesn't throw) when no adapter is found
      if (await tf.setBackend("webgpu").catch(() => false)) {
        await tf.ready();
        return "webgpu";
      }
    }
    console.warn("jskit-learn: WebGPU is not available, using cpu");
  }

  await tf.setBackend("cpu");
  await tf.ready();
  return "cpu";
};

/** {@link Backend} implemented with TensorFlow.js; runs on whichever tfjs backend is active. */
export const tfjsBackend: Backend = {
  get name() {
    return tf.getBackend() as BackendName;
  },

  matrix: (data) => fromTf(tf.tensor2d(data)),
  column: (data) => fromTf(tf.tensor2d(data, [data.length, 1])),
  zeros: (rows, cols) => fromTf(tf.zeros([rows, cols])),
  read: async (t) => Array.from(await toTf(t).data()),

  matmul: (a, b) => fromTf(tf.matMul(toTf(a), toTf(b))),
  add: (a, b) => fromTf(tf.add(toTf(a), toTf(b))),
  sub: (a, b) => fromTf(tf.sub(toTf(a), toTf(b))),
  scale: (a, scalar) => fromTf(tf.mul(toTf(a), scalar)),
  transpose: (a) => fromTf(tf.transpose(toTf(a))),
  abs: (a) => fromTf(tf.abs(toTf(a))),
  step: (a) => fromTf(tf.step(toTf(a))),
  // keepDims turns the 2D sum into a 1 × 1 matrix without an intermediate tensor
  sum: (a) => fromTf(tf.sum(toTf(a), undefined, true)),

  dispose: (t) => tf.dispose(toTf(t)),
};

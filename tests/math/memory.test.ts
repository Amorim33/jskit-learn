import { expect, test } from "bun:test";
import * as tf from "@tensorflow/tfjs-core";
import { matrix, sum, toArray } from "../../src/math";

// Finalizers run asynchronously after a collection, so collect a few times.
const collect = async () => {
  for (let i = 0; i < 5; i++) {
    Bun.gc(true);
    await Bun.sleep(10);
  }
};

test("backend memory is released once matrices are garbage-collected", async () => {
  // start from a clean slate: garbage left by other test files would be collected below too
  await collect();
  const before = tf.memory().numTensors;

  await (async () => {
    for (let i = 0; i < 100; i++) {
      await toArray(sum(matrix([[i, i + 1]])));
    }
  })();
  expect(tf.memory().numTensors).toBeGreaterThan(before);

  await collect();
  expect(tf.memory().numTensors).toBe(before);
});

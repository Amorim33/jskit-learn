import { describe, expect, test } from "bun:test";
import { Perceptron } from "../../src";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

describe("Perceptron", () => {
  test("uses default hyperparameters", () => {
    const p = new Perceptron({ inputSize: 2 });
    expect(p.inputSize).toBe(2);
    expect(p.learningRate).toBe(1);
  });

  test("predicts 0 before training (zero weights)", async () => {
    const p = new Perceptron({ inputSize: 2 });
    expect(await p.predictBatch(X)).toEqual([0, 0, 0, 0]);
  });

  test("learns the AND function", async () => {
    const y = [0, 0, 0, 1];
    const p = new Perceptron({ inputSize: 2 });
    await p.train(X, y, 100);
    expect(await p.predictBatch(X)).toEqual(y);
  });

  test("learns the OR function", async () => {
    const y = [0, 1, 1, 1];
    const p = new Perceptron({ inputSize: 2 });
    await p.train(X, y, 100);
    expect(await p.predictBatch(X)).toEqual(y);
  });

  test("does not learn XOR (not linearly separable)", async () => {
    const y = [0, 1, 1, 0];
    const p = new Perceptron({ inputSize: 2 });
    await p.train(X, y, 100);
    expect(await p.predictBatch(X)).not.toEqual(y);
  });

  test("predict classifies a single sample", async () => {
    const p = new Perceptron({ inputSize: 2 });
    await p.train(X, [0, 0, 0, 1], 100);
    expect(await p.predict([1, 1])).toBe(1);
    expect(await p.predict([0, 1])).toBe(0);
  });

  test("returns plain arrays and does not mutate inputs", async () => {
    const p = new Perceptron({ inputSize: 2 });
    const sample = [1, 0];
    const result = await p.predictBatch([sample]);
    expect(Array.isArray(result)).toBe(true);
    expect(sample).toEqual([1, 0]);
  });
});

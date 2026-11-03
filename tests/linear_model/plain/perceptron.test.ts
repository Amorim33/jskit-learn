import { describe, expect, test } from "bun:test";
import { Perceptron, PlainPerceptron } from "../../../src";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const predictAll = (p: PlainPerceptron) => X.map((x) => p.predict(x));

describe("PlainPerceptron", () => {
  test("uses default hyperparameters", () => {
    const p = new PlainPerceptron({ inputSize: 2 });
    expect(p.inputSize).toBe(2);
    expect(p.learningRate).toBe(1);
  });

  test("predicts 0 before training (zero weights)", () => {
    expect(predictAll(new PlainPerceptron({ inputSize: 2 }))).toEqual([0, 0, 0, 0]);
  });

  test("learns the AND function", () => {
    const y = [0, 0, 0, 1];
    const p = new PlainPerceptron({ inputSize: 2 });
    p.train(X, y, 100);
    expect(predictAll(p)).toEqual(y);
  });

  test("learns the OR function", () => {
    const y = [0, 1, 1, 1];
    const p = new PlainPerceptron({ inputSize: 2 });
    p.train(X, y, 100);
    expect(predictAll(p)).toEqual(y);
  });

  test("does not learn XOR (not linearly separable)", () => {
    const y = [0, 1, 1, 0];
    const p = new PlainPerceptron({ inputSize: 2 });
    p.train(X, y, 100);
    expect(predictAll(p)).not.toEqual(y);
  });

  test("does not mutate inputs", () => {
    const sample = [1, 0];
    new PlainPerceptron({ inputSize: 2 }).predict(sample);
    expect(sample).toEqual([1, 0]);
  });

  test("train and trainSteps reach the same weights", () => {
    const y = [0, 0, 0, 1];
    const a = new PlainPerceptron({ inputSize: 2 });
    a.train(X, y, 100);
    const b = new PlainPerceptron({ inputSize: 2 });
    for (const _ of b.trainSteps(X, y, 100)) {
      // consume every step
    }
    expect(b.weights).toEqual(a.weights);
  });

  test("trainSteps yields every sample, then the epoch with its error count", () => {
    const y = [0, 0, 0, 1];
    const steps = [...new PlainPerceptron({ inputSize: 2 }).trainSteps(X, y, 100)];
    const samples = steps.filter((s) => s.type === "sample");
    const epochs = steps.filter((s) => s.type === "epoch");

    expect(samples.length).toBe(X.length * epochs.length);
    // each epoch step reports how many of its samples had an error
    for (const e of epochs) {
      const wrong = samples.filter((s) => s.epoch === e.epoch && s.error !== 0).length;
      expect(e.errors).toBe(wrong);
    }
    // training stopped early, on the first epoch without errors
    expect(epochs.at(-1)?.errors).toBe(0);
    expect(epochs.length).toBeLessThan(100);
  });

  test("trainSteps reports the weighted sum and prediction before the update", () => {
    const [first] = new PlainPerceptron({ inputSize: 2 }).trainSteps(X, [1, 0, 0, 1], 1);
    // zero weights: w·x + b = 0, so the prediction is 0 and the error is 1
    expect(first).toEqual({
      type: "sample",
      epoch: 0,
      index: 0,
      x: [0, 0],
      weightedSum: 0,
      prediction: 0,
      target: 1,
      error: 1,
    });
  });

  test("weights returns a copy", () => {
    const p = new PlainPerceptron({ inputSize: 2 });
    p.weights[0] = 42;
    expect(p.weights).toEqual([0, 0, 0]);
  });

  test("agrees with the tensor Perceptron on separable data", async () => {
    // label = 1 when the sum of the features is positive
    const data = Array.from({ length: 200 }, (_, i) => [
      Math.sin(i),
      Math.cos(3 * i),
      Math.sin(7 * i),
    ]);
    const labels = data.map((x) => (x[0] + x[1] + x[2] > 0 ? 1 : 0));

    const plain = new PlainPerceptron({ inputSize: 3 });
    plain.train(data, labels, 1000);
    const tensor = new Perceptron({ inputSize: 3 });
    await tensor.train(data, labels, 1000);

    // both rules converge on separable data, so both reproduce the labels
    expect(data.map((x) => plain.predict(x))).toEqual(labels);
    expect(await tensor.predictBatch(data)).toEqual(labels);
  });
});

import { heavside, scalarProduct } from "../math/plain";
import type { PerceptronOptions } from "./perceptron-tensor";

/**
 * One step of {@link PlainPerceptron.trainSteps}: a processed sample, or the end of an epoch.
 */
export type PerceptronStep =
  | {
      type: "sample";
      /** Epoch this sample belongs to, from 0. */
      epoch: number;
      /** Index of the sample in `X`. */
      index: number;
      /** The sample, without the bias input. */
      x: number[];
      /** `w · x + b`, before the update. */
      weightedSum: number;
      /** `heavside(weightedSum)`: the predicted label. */
      prediction: number;
      /** The expected label. */
      target: number;
      /** `target - prediction`; the weights were updated iff it is not 0. */
      error: number;
    }
  | {
      type: "epoch";
      /** Epoch that just finished, from 0. */
      epoch: number;
      /** How many samples were misclassified during the epoch. */
      errors: number;
    };

/**
 * Rosenblatt's perceptron in plain JavaScript: the didactic version of `Perceptron`.
 *
 * @remarks
 * Uses online learning: weights are updated after each sample with
 * `w <- w + lr * (y - y_hat) * x`. The bias is stored as the last weight and
 * paired with a constant input of 1. Training stops after `epochs` passes or
 * as soon as an epoch has no errors.
 *
 * @example
 * ```ts
 * const p = new PlainPerceptron({ inputSize: 2 });
 * p.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1], 100);
 * p.predict([1, 1]); // 1
 * ```
 */
export class PlainPerceptron {
  /** Number of features in each input sample (excluding the bias). */
  readonly inputSize: number;
  /** Step size applied to each weight update. */
  readonly learningRate: number;

  /** Weights: one per feature, with the bias as the last entry. */
  private W: number[];

  /**
   * Creates a perceptron with all weights (and the bias) initialized to zero.
   *
   * @param options - See {@link PerceptronOptions}.
   */
  constructor({ inputSize, learningRate = 1 }: PerceptronOptions) {
    this.inputSize = inputSize;
    this.learningRate = learningRate;

    // one weight per feature, plus the bias
    this.W = new Array<number>(inputSize + 1).fill(0);
  }

  /**
   * Appends the constant bias input to a sample.
   *
   * @param x - A sample with {@link PlainPerceptron.inputSize} features.
   * @returns A new array `[...x, 1]`; `x` is not modified.
   */
  private prepareInput(x: number[]) {
    return [...x, 1];
  }

  /** A copy of the weights, one per feature, with the bias as the last entry. */
  get weights(): number[] {
    return [...this.W];
  }

  /**
   * Computes the weighted sum of a sample, bias included.
   *
   * @param augmentedX - A sample already extended with the bias input.
   * @returns `w · x + b`.
   */
  private weightedSum(augmentedX: number[]) {
    return scalarProduct(this.W, augmentedX);
  }

  /**
   * Computes the step activation of the weighted sum.
   *
   * @param augmentedX - A sample already extended with the bias input.
   * @returns `1` if the weighted sum is positive, otherwise `0`.
   */
  private activate(augmentedX: number[]) {
    return heavside(this.weightedSum(augmentedX));
  }

  /**
   * Runs one pass over the data, updating the weights after each misclassified sample.
   *
   * @param X - Training samples, one per row.
   * @param y - Target labels (`0` or `1`), aligned with `X`.
   * @param epoch - Index of this pass, reported in the yielded steps.
   * @returns How many samples were misclassified during this pass.
   */
  private *trainEpoch(
    X: number[][],
    y: number[],
    epoch: number,
  ): Generator<PerceptronStep, number> {
    let errors = 0;

    for (let i = 0; i < X.length; i++) {
      const x = this.prepareInput(X[i]);
      const weightedSum = this.weightedSum(x);
      const prediction = heavside(weightedSum);
      const error = y[i] - prediction;

      yield {
        type: "sample",
        epoch,
        index: i,
        x: X[i],
        weightedSum,
        prediction,
        target: y[i],
        error,
      };

      if (error !== 0) {
        errors++;
        for (let j = 0; j < x.length; j++) {
          this.W[j] = this.W[j] + this.learningRate * error * x[j];
        }
      }
    }

    return errors;
  }

  /**
   * Trains the perceptron step by step, so the caller controls the pace.
   *
   * @remarks
   * Each sample step is yielded before its update is applied: the update
   * happens when the caller asks for the next step. Stops early once an epoch
   * has no errors.
   *
   * @param X - Training samples, one per row.
   * @param y - Target labels (`0` or `1`), aligned with `X`.
   * @param epochs - Maximum number of passes over the data.
   * @returns A generator of {@link PerceptronStep}s.
   *
   * @example
   * ```ts
   * for (const step of p.trainSteps(X, y, 10)) {
   *   if (step.type === "epoch") console.log(step.epoch, step.errors);
   * }
   * ```
   */
  *trainSteps(X: number[][], y: number[], epochs = 1): Generator<PerceptronStep, void> {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const errors = yield* this.trainEpoch(X, y, epoch);
      yield { type: "epoch", epoch, errors };
      if (errors === 0) break;
    }
  }

  /**
   * Trains the perceptron, stopping early once an epoch has no errors.
   *
   * @param X - Training samples, one per row.
   * @param y - Target labels (`0` or `1`), aligned with `X`.
   * @param epochs - Maximum number of passes over the data.
   */
  train(X: number[][], y: number[], epochs = 1) {
    for (const _ of this.trainSteps(X, y, epochs)) {
      // consume every step
    }
  }

  /**
   * Classifies a single sample.
   *
   * @param x - A sample with {@link PlainPerceptron.inputSize} features.
   * @returns The predicted label, `0` or `1`.
   * @throws Error if `x` does not have {@link PlainPerceptron.inputSize} features.
   */
  predict(x: number[]): number {
    return this.activate(this.prepareInput(x));
  }
}

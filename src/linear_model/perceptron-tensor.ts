import {
  abs,
  add,
  column,
  heavside,
  type Matrix,
  matmul,
  matrix,
  scale,
  sub,
  sum,
  toArray,
  transpose,
  zeros,
} from "../math";
import { perceptronSvg, type Row } from "../plot";

/**
 * Options for {@link Perceptron.draw}.
 */
export interface DrawOptions {
  /** Title drawn above the figure. */
  title?: string;
}

/** What the last call to {@link Perceptron.train} did. */
interface TrainingStats {
  epochs: number;
  maxEpochs: number;
  samples: number;
}

/**
 * Options for creating a {@link Perceptron}.
 */
export interface PerceptronOptions {
  /** Number of features in each input sample (excluding the bias). */
  inputSize: number;

  /**
   * Step size applied to each weight update.
   * @defaultValue 1
   */
  learningRate?: number;
}

/**
 * Rosenblatt's perceptron: a binary linear classifier.
 *
 * @remarks
 * Uses the batch rule, so a whole epoch is a few matrix operations:
 * `w <- w + lr * Xᵀ(y - y_hat)` and `b <- b + lr * Σ(y - y_hat)`.
 * Training stops after `epochs` passes or as soon as an epoch has no errors.
 *
 * @example
 * ```ts
 * const p = new Perceptron({ inputSize: 2 });
 * await p.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1], 100);
 * await p.predict([1, 1]); // 1
 * ```
 */
export class Perceptron {
  readonly inputSize: number;
  readonly learningRate: number;

  private W: Matrix;
  private b: Matrix;
  private stats: TrainingStats | null = null;

  /**
   * Creates a perceptron with all weights (and the bias) initialized to zero.
   *
   * @param options - See {@link PerceptronOptions}.
   */
  constructor({ inputSize, learningRate = 1 }: PerceptronOptions) {
    this.inputSize = inputSize;
    this.learningRate = learningRate;

    this.W = zeros(inputSize, 1);
    this.b = zeros(1, 1);
  }

  /**
   * Computes the step activation of `X · W + b`.
   *
   * @param X - Samples as an `n × inputSize` matrix.
   * @returns Predicted labels as an `n × 1` column of `0`/`1`.
   */
  private activate(X: Matrix) {
    return heavside(add(matmul(X, this.W), this.b));
  }

  /**
   * Runs one pass over the data with the batch update rule.
   *
   * @param X - Training samples as an `n × inputSize` matrix.
   * @param y - Target labels as an `n × 1` column.
   * @returns How many samples were misclassified before the update.
   */
  private async trainEpoch(X: Matrix, y: Matrix) {
    const error = sub(y, this.activate(X));

    this.W = add(this.W, scale(matmul(transpose(X), error), this.learningRate));
    this.b = add(this.b, scale(sum(error), this.learningRate));

    const [errors = 0] = await toArray(sum(abs(error)));
    return errors;
  }

  /**
   * Trains the perceptron, stopping early once an epoch has no errors.
   *
   * @param X - Training samples, one per row.
   * @param y - Target labels (`0` or `1`), aligned with `X`.
   * @param epochs - Maximum number of passes over the data.
   */
  async train(X: number[][], y: number[], epochs = 1): Promise<void> {
    const [Xm, ym] = [matrix(X), column(y)];
    const stats: TrainingStats = { epochs: 0, maxEpochs: epochs, samples: X.length };
    this.stats = stats;

    while (stats.epochs < epochs) {
      stats.epochs++;
      if ((await this.trainEpoch(Xm, ym)) === 0) break;
    }
  }

  /**
   * Classifies many samples at once.
   *
   * @param X - Samples, one per row, each with {@link Perceptron.inputSize} features.
   * @returns The predicted labels (`0` or `1`), one per sample.
   */
  predictBatch(X: number[][]): Promise<number[]> {
    return toArray(this.activate(matrix(X)));
  }

  /**
   * Classifies a single sample.
   *
   * @param x - A sample with {@link Perceptron.inputSize} features.
   * @returns The predicted label, `0` or `1`.
   */
  async predict(x: number[]): Promise<number> {
    const [prediction = 0] = await this.predictBatch([x]);
    return prediction;
  }

  /**
   * Writes a paper-ready SVG diagram of the model: the inputs and bias feeding Σ
   * through the learned weights, the step activation, and tables with the
   * hyperparameters and the statistics of the last {@link Perceptron.train} call.
   *
   * @param path - Output file; must end in `.svg`.
   * @param options - See {@link DrawOptions}.
   *
   * @example
   * ```ts
   * await p.train(X, y, 100);
   * await p.draw("perceptron.svg", { title: "Perceptron learning AND" });
   * ```
   */
  async draw(path: string, options: DrawOptions = {}): Promise<void> {
    if (!path.toLowerCase().endsWith(".svg")) {
      throw new Error(
        `Perceptron.draw writes SVG; expected a path ending in ".svg", got "${path}"`,
      );
    }

    const [weights, [bias = 0]] = await Promise.all([toArray(this.W), toArray(this.b)]);

    const hyperparameters: Row[] = [
      ["inputs (n)", String(this.inputSize)],
      ["learning rate (η)", String(this.learningRate)],
      ["activation", "Heaviside step"],
      ["update rule", "w ← w + η Xᵀ(y − ŷ)"],
      ["initialization", "zeros"],
    ];

    const s = this.stats;
    const training: Row[] = s
      ? [
          ["epochs", `${s.epochs} / ${s.maxEpochs}`],
          ["samples", String(s.samples)],
        ]
      : [["status", "not trained"]];

    const svg = perceptronSvg({
      weights,
      bias,
      hyperparameters,
      training,
      ...(options.title === undefined ? {} : { title: options.title }),
    });

    // Imported lazily so browser bundles that never draw don't pull in `node:fs`.
    const { writeFile } = await import("node:fs/promises");
    await writeFile(path, svg, "utf8");
  }
}

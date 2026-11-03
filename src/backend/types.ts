/** Compute backends the library can run on. */
export type BackendName = "cpu" | "webgpu";

/**
 * Opaque handle to data owned by a {@link Backend}.
 *
 * @remarks
 * Internal only: models work with `Matrix` (from `src/math`), and the public API
 * takes and returns plain arrays.
 */
export interface Tensor {
  readonly __tensor: unique symbol;
}

/**
 * Tensor operations the math module is built on.
 *
 * @remarks
 * Keeps the rest of the library independent from the engine underneath
 * (currently TensorFlow.js), so it can be swapped for hand-written kernels later.
 * Grows on demand.
 */
export interface Backend {
  /** Which backend is currently running the operations. */
  readonly name: BackendName;

  /**
   * Uploads a matrix.
   *
   * @param data - Rows of equal length.
   */
  matrix(data: number[][]): Tensor;

  /**
   * Uploads a vector as an `n × 1` column matrix.
   *
   * @param data - Vector values.
   */
  column(data: number[]): Tensor;

  /**
   * Creates a `rows × cols` matrix filled with zeros.
   *
   * @param rows - Number of rows.
   * @param cols - Number of columns.
   */
  zeros(rows: number, cols: number): Tensor;

  /**
   * Downloads a tensor as a flat array (row-major).
   *
   * @param t - Tensor to read.
   * @returns A plain `number[]`.
   */
  read(t: Tensor): Promise<number[]>;

  /** Matrix product `a · b`. */
  matmul(a: Tensor, b: Tensor): Tensor;
  /** Element-wise `a + b`, with broadcasting. */
  add(a: Tensor, b: Tensor): Tensor;
  /** Element-wise `a - b`, with broadcasting. */
  sub(a: Tensor, b: Tensor): Tensor;
  /** Multiplies every element by `scalar`. */
  scale(a: Tensor, scalar: number): Tensor;
  /** Matrix transpose. */
  transpose(a: Tensor): Tensor;
  /** Element-wise absolute value. */
  abs(a: Tensor): Tensor;
  /** Element-wise Heaviside step: `x > 0 ? 1 : 0`. */
  step(a: Tensor): Tensor;
  /** Sum of all elements, as a `1 × 1` matrix. */
  sum(a: Tensor): Tensor;

  /**
   * Frees the memory held by a tensor.
   *
   * @param t - Tensor to release.
   */
  dispose(t: Tensor): void;
}

import type { Matrix } from "./matrix";
import { matmul, transpose } from "./operations";

/**
 * Computes the scalar (dot) product of two column vectors.
 *
 * @param X1 - First vector, as an `n × 1` matrix.
 * @param X2 - Second vector, as an `n × 1` matrix.
 * @returns A `1 × 1` matrix with the sum of `X1[i] * X2[i]`.
 * @throws Error if the vectors have different lengths.
 */
export const scalarProduct = (X1: Matrix, X2: Matrix) => matmul(transpose(X1), X2);

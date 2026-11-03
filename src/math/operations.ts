import { backend } from "../backend";
import { type Matrix, unwrap, wrap } from "./matrix";

/**
 * Matrix product.
 *
 * @param a - An `n × k` matrix.
 * @param b - A `k × m` matrix.
 * @returns The `n × m` matrix `a · b`.
 */
export const matmul = (a: Matrix, b: Matrix) => wrap(backend.matmul(unwrap(a), unwrap(b)));

/**
 * Element-wise sum. A `1 × 1` operand is broadcast to the other's shape.
 *
 * @param a - First matrix.
 * @param b - Second matrix.
 * @returns `a + b`.
 */
export const add = (a: Matrix, b: Matrix) => wrap(backend.add(unwrap(a), unwrap(b)));

/**
 * Element-wise difference. A `1 × 1` operand is broadcast to the other's shape.
 *
 * @param a - First matrix.
 * @param b - Second matrix.
 * @returns `a - b`.
 */
export const sub = (a: Matrix, b: Matrix) => wrap(backend.sub(unwrap(a), unwrap(b)));

/**
 * Multiplies every element by a scalar.
 *
 * @param a - A matrix.
 * @param scalar - Factor.
 * @returns `scalar * a`.
 */
export const scale = (a: Matrix, scalar: number) => wrap(backend.scale(unwrap(a), scalar));

/**
 * Matrix transpose.
 *
 * @param a - An `n × m` matrix.
 * @returns The `m × n` matrix `aᵀ`.
 */
export const transpose = (a: Matrix) => wrap(backend.transpose(unwrap(a)));

/**
 * Element-wise absolute value.
 *
 * @param a - A matrix.
 * @returns `|a|`.
 */
export const abs = (a: Matrix) => wrap(backend.abs(unwrap(a)));

/**
 * Sum of all elements.
 *
 * @param a - A matrix.
 * @returns A `1 × 1` matrix with the sum.
 */
export const sum = (a: Matrix) => wrap(backend.sum(unwrap(a)));

import { backend } from "../backend";
import { type Matrix, unwrap, wrap } from "./matrix";

/**
 * Heaviside step function, applied element-wise.
 *
 * @param input - Matrix to evaluate.
 * @returns A matrix with `1` where `input` is positive, otherwise `0` (so `0` maps to `0`).
 */
export const heavside = (input: Matrix) => wrap(backend.step(unwrap(input)));

/**
 * Heaviside step function.
 *
 * @param input - Value to evaluate.
 * @returns `1` if `input` is positive, otherwise `0` (so `heavside(0) === 0`).
 */
export const heavside = (input: number) => {
  return input <= 0 ? 0 : 1;
};

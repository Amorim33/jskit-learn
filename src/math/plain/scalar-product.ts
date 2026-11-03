/**
 * Computes the scalar (dot) product of two vectors.
 *
 * @param X1 - First vector.
 * @param X2 - Second vector, with the same length as `X1`.
 * @returns The sum of `X1[i] * X2[i]` over all indices.
 * @throws Error if the vectors have different lengths.
 */
export const scalarProduct = (X1: number[], X2: number[]) => {
  if (X1.length !== X2.length) {
    throw new Error("Different input lenghts");
  }

  let result = 0;
  for (let i = 0; i < X1.length; i++) {
    result += X1[i] * X2[i];
  }

  return result;
};

import { describe, expect, test } from "bun:test";
import { heavside, scalarProduct } from "../../../src/math/plain";

describe("plain math", () => {
  test("scalarProduct", () => {
    // 2 - 0.9375 - 6 + 3
    expect(scalarProduct([0.5, -1.25, 3, 2], [4, 0.75, -2, 1.5])).toBeCloseTo(-1.9375, 10);
  });

  test("scalarProduct rejects vectors of different lengths", () => {
    expect(() => scalarProduct([1, 2], [1])).toThrow("Different input lenghts");
  });

  test("heavside maps positives to 1 and the rest (including 0) to 0", () => {
    expect([-2, -0.5, 0, 0.5, 2].map(heavside)).toEqual([0, 0, 0, 1, 1]);
  });
});

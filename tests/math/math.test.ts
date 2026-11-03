import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { setBackend } from "../../src";
import {
  abs,
  add,
  column,
  heavside,
  matmul,
  matrix,
  scalarProduct,
  scale,
  sub,
  sum,
  toArray,
  transpose,
  zeros,
} from "../../src/math";

// The same math must give the same results on every backend.
const gpuAvailable = (await setBackend("webgpu")) === "webgpu";

// the active backend is global: leave it on cpu for the other test files
afterAll(() => setBackend("cpu"));

for (const name of ["cpu", "webgpu"] as const) {
  describe.skipIf(name === "webgpu" && !gpuAvailable)(`math on ${name}`, () => {
    beforeAll(async () => {
      expect(await setBackend(name)).toBe(name);
    });

    test("matrix, column and zeros round-trip through toArray", async () => {
      expect(
        await toArray(
          matrix([
            [1, 2],
            [3, 4],
          ]),
        ),
      ).toEqual([1, 2, 3, 4]);
      expect(await toArray(column([5, 6]))).toEqual([5, 6]);
      expect(await toArray(zeros(2, 3))).toEqual([0, 0, 0, 0, 0, 0]);
    });

    test("scalarProduct", async () => {
      const [value] = await toArray(
        scalarProduct(column([0.5, -1.25, 3, 2]), column([4, 0.75, -2, 1.5])),
      );
      // 2 - 0.9375 - 6 + 3
      expect(value).toBeCloseTo(-1.9375, 5);
    });

    test("heavside maps positives to 1 and the rest (including 0) to 0", async () => {
      expect(await toArray(heavside(column([-2, -0.5, 0, 0.5, 2])))).toEqual([0, 0, 0, 1, 1]);
    });

    test("matmul and transpose", async () => {
      const m = matrix([
        [1, 2],
        [3, 4],
      ]);
      expect(await toArray(transpose(m))).toEqual([1, 3, 2, 4]);
      expect(await toArray(matmul(m, column([1, -1])))).toEqual([-1, -1]);
    });

    test("add and sub broadcast a 1 × 1 operand", async () => {
      const m = column([1, 2, 3]);
      const one = matrix([[1]]);
      expect(await toArray(add(m, one))).toEqual([2, 3, 4]);
      expect(await toArray(sub(m, one))).toEqual([0, 1, 2]);
    });

    test("scale, abs and sum", async () => {
      expect(await toArray(sum(abs(scale(column([1, -2, 3]), 2))))).toEqual([12]);
    });
  });
}

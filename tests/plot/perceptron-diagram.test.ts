import { afterAll, describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { formatNumber, Perceptron, perceptronSvg } from "../../src";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const dir = await mkdtemp(join(tmpdir(), "jskit-draw-"));
afterAll(() => rm(dir, { recursive: true, force: true }));

describe("formatNumber", () => {
  test("uses 3 decimals and a typographic minus", () => {
    expect(formatNumber(0.7314)).toBe("0.731");
    expect(formatNumber(-1)).toBe("−1.000");
    expect(formatNumber(-0)).toBe("0.000");
    expect(formatNumber(-0.0001)).toBe("0.000");
  });
});

describe("perceptronSvg", () => {
  const spec = {
    weights: [0.5, -1.25],
    bias: -2,
    hyperparameters: [["learning rate (η)", "0.1"]] as const,
    training: [["samples", "4"]] as const,
  };

  test("draws every weight, the bias, Σ and both tables", () => {
    const svg = perceptronSvg(spec);
    expect(svg.startsWith("<?xml")).toBe(true);
    expect(svg).toContain("<svg");
    expect(svg).toContain("0.500");
    expect(svg).toContain("−1.250");
    expect(svg).toContain("−2.000");
    expect(svg).toContain("Σ");
    expect(svg).toContain("Hyperparameters");
    expect(svg).toContain("learning rate (η)");
    expect(svg).toContain("Training");
    expect(svg).toContain("samples");
  });

  test("hides the training table when null and escapes the title", () => {
    const svg = perceptronSvg({ ...spec, training: null, title: "A < B & C" });
    expect(svg).not.toContain(">Training<");
    expect(svg).toContain("A &lt; B &amp; C");
  });

  test("elides the middle inputs of wide models", () => {
    const weights = Array.from({ length: 20 }, (_, i) => i / 100);
    const svg = perceptronSvg({ ...spec, weights });
    expect(svg).toContain("⋮");
    expect(svg.match(/class="node"/g)?.length).toBe(4 + 2); // x₁ x₂ x₃ xₙ, Σ, box
    expect(svg).toContain(">n<");
    expect(svg).toContain("0.190"); // last weight
    expect(svg).not.toContain("0.100"); // an elided one
  });
});

describe("Perceptron.draw", () => {
  test("writes an SVG with the learned weights and training stats", async () => {
    const p = new Perceptron({ inputSize: 2, learningRate: 0.5 });
    await p.train(X, [0, 0, 0, 1], 100);
    const path = join(dir, "and.svg");
    await p.draw(path, { title: "AND" });

    const svg = await readFile(path, "utf8");
    expect(svg).toContain("<svg");
    expect(svg).toContain(">AND<");
    expect(svg).toContain(">0.5<"); // learning rate

    const epochs = svg.match(/>(\d+) \/ 100</);
    expect(epochs).not.toBeNull();
    expect(Number(epochs?.[1])).toBeLessThan(100);
    expect(svg).toMatch(/>samples<\/text>\s*<text[^>]*>4</);
    expect(svg).not.toContain("converged");
  });

  test("reports an untrained model", async () => {
    const path = join(dir, "untrained.svg");
    await new Perceptron({ inputSize: 3 }).draw(path);
    expect(await readFile(path, "utf8")).toContain("not trained");
  });

  test("rejects non-SVG paths", async () => {
    const p = new Perceptron({ inputSize: 2 });
    await expect(p.draw(join(dir, "fig.png"))).rejects.toThrow(".svg");
  });
});

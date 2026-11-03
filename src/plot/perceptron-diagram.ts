/** A `[label, value]` row of a table in the figure. */
export type Row = readonly [label: string, value: string];

/**
 * Everything drawn by {@link perceptronSvg}.
 */
export interface PerceptronDiagramSpec {
  /** Learned weights, one per input feature. */
  weights: readonly number[];
  /** Learned bias. */
  bias: number;
  /** Rows of the "Hyperparameters" table. */
  hyperparameters: readonly Row[];
  /** Rows of the "Training" table; `null` hides the table. */
  training: readonly Row[] | null;
  /** Optional title drawn above the figure. */
  title?: string;
}

/** Every size in the figure, in SVG user units (≈ px at 100% zoom). */
const layout = {
  pad: 28,
  font: 15,
  sub: 11,
  title: 17,
  node: 17,
  rowGap: 52,
  sum: 24,
  edge: 230,
  arrow: 52,
  box: { w: 58, h: 44 },
  output: 44,
  tableGap: 48,
  tableRow: 22,
  tableSection: 18,
  colGap: 28,
  /** Inputs drawn before eliding the middle ones. */
  maxInputs: 8,
  ink: "#111",
  muted: "#666",
  fill: "#f2f2f2",
} as const;

const fontFamily = `"Latin Modern Roman", "CMU Serif", "Times New Roman", Times, serif`;

/** Escapes text for use inside SVG. */
const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Formats a number with 3 decimals and a typographic minus sign. */
export const formatNumber = (v: number) => {
  const s = (Object.is(v, -0) ? 0 : v).toFixed(3);
  return s === "-0.000" ? "0.000" : s.replace("-", "−");
};

/** Rough width of serif text; good enough to size table columns. */
const textWidth = (s: string, size: number) => [...s].length * size * 0.52;

/** Rounds a coordinate to keep the markup short. */
const r = (v: number) => Math.round(v * 100) / 100;

/** An italic symbol with a subscript (italic too when it's a letter), as SVG `<tspan>`s. */
const subscripted = (symbol: string, index: string) => {
  const cls = /^\d+$/.test(index) ? "" : ` class="m"`;
  return `<tspan class="m">${symbol}</tspan><tspan${cls} dy="4" font-size="${layout.sub}">${esc(index)}</tspan><tspan dy="-4">​</tspan>`;
};

type InputRow = { kind: "x"; index: number } | { kind: "dots" } | { kind: "bias" };

/** Which input rows to draw: all of them, or the first three, an ellipsis and the last. */
const inputRows = (n: number): InputRow[] => {
  const shown: number[] =
    n <= layout.maxInputs ? Array.from({ length: n }, (_, i) => i) : [0, 1, 2, -1, n - 1];
  const rows: InputRow[] = shown.map((i) =>
    i === -1 ? { kind: "dots" } : { kind: "x", index: i },
  );
  rows.push({ kind: "bias" });
  return rows;
};

type Section = readonly [caption: string, rows: readonly Row[]];

/** Column widths shared by all tables, so stacked tables line up. */
const tableColumns = (sections: readonly Section[]) => {
  const w = (s: string) => textWidth(s, layout.font);
  const labelW = Math.max(...sections.flatMap(([c, rows]) => [w(c), ...rows.map(([l]) => w(l))]));
  const valueW = Math.max(...sections.flatMap(([, rows]) => rows.map(([, v]) => w(v))));
  return { labelW, width: labelW + layout.colGap + valueW };
};

/** Draws a booktabs-style table with a bold caption; returns its markup and height. */
const table = (
  x: number,
  y: number,
  [caption, rows]: Section,
  { labelW, width }: ReturnType<typeof tableColumns>,
) => {
  const { tableRow, colGap } = layout;
  const parts: string[] = [];
  let cy = y;

  parts.push(`<line class="rule-thick" x1="${x}" y1="${cy}" x2="${x + width}" y2="${cy}"/>`);
  cy += tableRow - 4;
  parts.push(`<text class="b" x="${x}" y="${cy}">${esc(caption)}</text>`);
  cy += 8;
  parts.push(`<line class="rule-thin" x1="${x}" y1="${cy}" x2="${x + width}" y2="${cy}"/>`);
  for (const [label, value] of rows) {
    cy += tableRow;
    parts.push(
      `<text x="${x}" y="${cy}">${esc(label)}</text>`,
      `<text x="${x + labelW + colGap}" y="${cy}">${esc(value)}</text>`,
    );
  }
  cy += 10;
  parts.push(`<line class="rule-thick" x1="${x}" y1="${cy}" x2="${x + width}" y2="${cy}"/>`);

  return { svg: parts.join("\n"), height: cy - y };
};

/** Height of {@link table} for `rows`, without drawing it. */
const tableHeight = (rows: number) => layout.tableRow - 4 + 8 + rows * layout.tableRow + 10;

/**
 * Renders a perceptron as a standalone, paper-ready SVG: inputs and bias feeding a
 * summation node through labelled weights, a step activation and the output, next to
 * tables of hyperparameters and training statistics.
 *
 * @param spec - What to draw; see {@link PerceptronDiagramSpec}.
 * @returns The SVG document as a string.
 */
export function perceptronSvg(spec: PerceptronDiagramSpec): string {
  const L = layout;
  const n = spec.weights.length;
  const rows = inputRows(n);

  const titleH = spec.title ? L.title + 16 : 0;
  const colH = (rows.length - 1) * L.rowGap + 2 * L.node;

  const sections: Section[] = [["Hyperparameters", spec.hyperparameters]];
  if (spec.training) sections.push(["Training", spec.training]);
  const tablesH =
    sections.reduce((h, [, r]) => h + tableHeight(r.length), 0) +
    (sections.length - 1) * L.tableSection;
  const columns = tableColumns(sections);
  const tablesW = columns.width;

  const contentH = Math.max(colH, tablesH);
  const top = L.pad + titleH;

  // Network geometry.
  const inX = L.pad + L.node + 6;
  const colTop = top + (contentH - colH) / 2 + L.node;
  const rowY = (i: number) => colTop + i * L.rowGap;
  const sumX = inX + L.edge;
  const sumY = colTop + ((rows.length - 1) * L.rowGap) / 2;
  const boxX = sumX + L.sum + L.arrow;
  const boxY = sumY - L.box.h / 2;
  const outX = boxX + L.box.w + L.output;
  const netRight = outX + 22;

  const tablesX = netRight + L.tableGap;
  const width = Math.ceil(tablesX + tablesW + L.pad);
  const height = Math.ceil(top + contentH + L.pad);

  const parts: string[] = [];

  if (spec.title) {
    parts.push(
      `<text class="title" x="${width / 2}" y="${L.pad + L.title}" text-anchor="middle">${esc(spec.title)}</text>`,
    );
  }

  // Inputs, bias and their weighted edges into Σ.
  rows.forEach((row, i) => {
    const y = rowY(i);
    if (row.kind === "dots") {
      parts.push(`<text class="m" x="${inX}" y="${y + 5}" text-anchor="middle">⋮</text>`);
      return;
    }

    const isBias = row.kind === "bias";
    const weight = isBias ? spec.bias : (spec.weights[row.index] ?? 0);
    const nodeLabel = isBias
      ? `<tspan>1</tspan>`
      : subscripted("x", row.index === n - 1 && n > L.maxInputs ? "n" : String(row.index + 1));
    const weightName = isBias
      ? `<tspan class="m">b</tspan>`
      : subscripted("w", row.index === n - 1 && n > L.maxInputs ? "n" : String(row.index + 1));

    // Edge from the node's border to Σ's border, leaving room for the arrowhead.
    const dx = sumX - inX;
    const dy = sumY - y;
    const len = Math.hypot(dx, dy);
    const [ux, uy] = [dx / len, dy / len];
    const x1 = inX + ux * L.node;
    const y1 = y + uy * L.node;
    const x2 = sumX - ux * (L.sum + 2);
    const y2 = sumY - uy * (L.sum + 2);

    // Label rides just above the edge, rotated to follow it.
    const t = 0.4;
    const lx = x1 + (x2 - x1) * t;
    const ly = y1 + (y2 - y1) * t;
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;

    parts.push(
      `<line class="edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#arrow)"/>`,
      `<circle class="node${isBias ? " bias" : ""}" cx="${inX}" cy="${y}" r="${L.node}"/>`,
      `<text x="${inX}" y="${y + 5}" text-anchor="middle">${nodeLabel}</text>`,
      `<text class="wlabel" transform="translate(${r(lx)} ${r(ly)}) rotate(${r(angle)})" y="-8" text-anchor="middle">${weightName} = ${formatNumber(weight)}</text>`,
    );
  });

  // Σ → step activation → ŷ.
  const stepL = boxX + 10;
  const stepR = boxX + L.box.w - 10;
  const stepM = (stepL + stepR) / 2;
  const lo = sumY + 9;
  const hi = sumY - 9;
  parts.push(
    `<circle class="node" cx="${sumX}" cy="${sumY}" r="${L.sum}"/>`,
    `<text class="big" x="${sumX}" y="${sumY + 7}" text-anchor="middle">Σ</text>`,
    `<line class="edge" x1="${sumX + L.sum}" y1="${sumY}" x2="${boxX - 2}" y2="${sumY}" marker-end="url(#arrow)"/>`,
    `<text class="m" x="${sumX + L.sum + L.arrow / 2}" y="${sumY - 8}" text-anchor="middle">z</text>`,
    `<rect class="node" x="${boxX}" y="${boxY}" width="${L.box.w}" height="${L.box.h}" rx="3"/>`,
    `<line class="axis" x1="${stepL}" y1="${lo}" x2="${stepR}" y2="${lo}"/>`,
    `<polyline class="step" points="${stepL},${lo} ${stepM},${lo} ${stepM},${hi} ${stepR},${hi}"/>`,
    `<text class="cap" x="${boxX + L.box.w / 2}" y="${boxY + L.box.h + 16}" text-anchor="middle"><tspan class="m">H</tspan>(<tspan class="m">z</tspan>)</text>`,
    `<line class="edge" x1="${boxX + L.box.w}" y1="${sumY}" x2="${outX - 2}" y2="${sumY}" marker-end="url(#arrow)"/>`,
    `<text class="m" x="${outX + 4}" y="${sumY + 5}">ŷ</text>`,
  );

  // Tables, vertically centred against the network.
  let ty = top + (contentH - tablesH) / 2;
  for (const section of sections) {
    const t = table(tablesX, ty, section, columns);
    parts.push(t.svg);
    ty += t.height + L.tableSection;
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family='${fontFamily}' font-size="${L.font}">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,1 L9,5 L0,9 z" fill="${L.ink}"/>
</marker>
<style>
text { fill: ${L.ink}; }
.m { font-style: italic; }
.b { font-weight: bold; }
.big { font-size: ${L.font + 6}px; }
.title { font-size: ${L.title}px; font-weight: bold; }
.cap { fill: ${L.muted}; font-size: ${L.font - 2}px; }
.wlabel { font-size: ${L.font - 1}px; }
.node { fill: #fff; stroke: ${L.ink}; stroke-width: 1.2; }
.bias { fill: ${L.fill}; }
.edge { stroke: ${L.ink}; stroke-width: 1; }
.axis { stroke: ${L.muted}; stroke-width: 0.6; }
.step { fill: none; stroke: ${L.ink}; stroke-width: 1.6; }
.rule-thick { stroke: ${L.ink}; stroke-width: 1.2; }
.rule-thin { stroke: ${L.ink}; stroke-width: 0.6; }
</style>
</defs>
<rect width="100%" height="100%" fill="#fff"/>
${parts.join("\n")}
</svg>
`;
}

import { backend, type Tensor } from "../backend";

/** Frees a matrix's backend memory once the matrix is garbage-collected. */
const registry = new FinalizationRegistry<Tensor>((tensor) => backend.dispose(tensor));
const tensors = new WeakMap<Matrix, Tensor>();

/**
 * A matrix living on the active backend (CPU or GPU).
 *
 * @remarks
 * Created with {@link matrix}, {@link column} or {@link zeros}, combined with the
 * operations in `src/math`, and read back with {@link toArray}. Its memory is
 * released automatically when it is no longer referenced.
 */
export class Matrix {
  /**
   * Wraps backend data. Internal: use {@link matrix}, {@link column} or {@link zeros}.
   *
   * @param tensor - Backend data owned by this matrix from now on.
   */
  constructor(tensor: Tensor) {
    tensors.set(this, tensor);
    registry.register(this, tensor);
  }
}

/**
 * Wraps backend data in a {@link Matrix}. Internal to `src/math`.
 *
 * @param tensor - Backend data.
 * @returns A matrix owning `tensor`.
 */
export const wrap = (tensor: Tensor) => new Matrix(tensor);

/**
 * Gets the backend data of a {@link Matrix}. Internal to `src/math`.
 *
 * @param m - A matrix.
 * @returns The backend data it owns.
 */
export const unwrap = (m: Matrix) => tensors.get(m) as Tensor;

/**
 * Creates a matrix from rows.
 *
 * @param rows - Rows of equal length.
 * @returns An `rows.length × rows[0].length` matrix.
 */
export const matrix = (rows: number[][]) => wrap(backend.matrix(rows));

/**
 * Creates a column matrix from a vector.
 *
 * @param values - Vector values.
 * @returns A `values.length × 1` matrix.
 */
export const column = (values: number[]) => wrap(backend.column(values));

/**
 * Creates a matrix filled with zeros.
 *
 * @param rows - Number of rows.
 * @param cols - Number of columns.
 * @returns A `rows × cols` matrix of zeros.
 */
export const zeros = (rows: number, cols: number) => wrap(backend.zeros(rows, cols));

/**
 * Reads a matrix back into a plain array.
 *
 * @param m - Matrix to read.
 * @returns Its values, flattened row by row.
 */
export const toArray = async (m: Matrix) => {
  const values = await backend.read(unwrap(m));
  // Referencing `m` after the await keeps it (and so its data) from being
  // collected and freed while the read is still in flight.
  tensors.get(m);
  return values;
};

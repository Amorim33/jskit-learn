import { activateTfjsBackend, tfjsBackend } from "./tfjs";
import type { Backend, BackendName } from "./types";

export type { Backend, BackendName, Tensor } from "./types";

/** The backend `src/math` runs on. */
export const backend: Backend = tfjsBackend;

/**
 * Selects where computations run. Defaults to `cpu` if never called.
 *
 * @remarks
 * Best called before creating models, so their parameters are allocated on the
 * chosen backend from the start.
 *
 * @param name - `"cpu"` or `"webgpu"`.
 * @returns The backend actually in use; `"cpu"` if WebGPU is unavailable.
 *
 * @example
 * ```ts
 * await setBackend("webgpu");
 * ```
 */
export const setBackend = (name: BackendName): Promise<BackendName> => activateTfjsBackend(name);

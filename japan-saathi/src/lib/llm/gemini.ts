import { ApiError, GoogleGenAI } from "@google/genai";

/**
 * Gemini client tuned for the FREE tier: low RPM quotas and frequent
 * "high demand" 503s. Every call goes through `withGemini`, which
 *   1. self-throttles per model to stay under the RPM quota,
 *   2. on 503/500 (overloaded) moves straight to the next model in the chain,
 *   3. on 429 waits the server's retryDelay (if short) and retries, else moves on,
 *   4. reports waits to the UI so a slow answer never looks like a hang.
 */
export const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

/** Fastest-first. Override with GEMINI_MODELS="a,b,c". */
export const GEMINI_MODELS = (process.env.GEMINI_MODELS ?? "gemini-3.5-flash,gemini-3.8-flash,gemini-flash-lite-latest")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const RPM = Number(process.env.GEMINI_RPM ?? 10);
const MAX_WAIT_MS = 25_000;

// ── Per-model sliding-window limiter ─────────────────────────────────────
const calls = new Map<string, number[]>();

function msUntilSlot(model: string, now = Date.now()) {
  const recent = (calls.get(model) ?? []).filter((t) => now - t < 60_000);
  calls.set(model, recent);
  return recent.length < RPM ? 0 : 60_000 - (now - recent[0]) + 50;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(signal.reason);
    });
  });

/** Pull "retryDelay": "23s" out of a 429 body. */
function retryDelayMs(err: ApiError): number | null {
  const m = err.message.match(/"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"/);
  return m ? Math.ceil(parseFloat(m[1]) * 1000) : null;
}

export class AllModelsBusyError extends Error {}

export async function withGemini<T>(
  fn: (model: string) => Promise<T>,
  opts: { onWait?: (message: string) => void; onSwitch?: (from: string, to: string) => void; signal?: AbortSignal } = {},
): Promise<{ result: T; model: string }> {
  let lastErr: unknown;
  for (let i = 0; i < GEMINI_MODELS.length; i++) {
    const model = GEMINI_MODELS[i];
    for (let attempt = 0; attempt < 2; attempt++) {
      const wait = msUntilSlot(model);
      if (wait > 0) {
        if (wait > MAX_WAIT_MS) break; // this model is saturated; try the next one
        opts.onWait?.(`Free-tier limit reached, continuing in ${Math.ceil(wait / 1000)}s…`);
        await sleep(wait, opts.signal);
      }
      calls.get(model)!.push(Date.now());
      try {
        return { result: await fn(model), model };
      } catch (err) {
        lastErr = err;
        if (!(err instanceof ApiError)) throw err;
        if (err.status === 429) {
          const d = retryDelayMs(err);
          if (d !== null && d <= MAX_WAIT_MS && attempt === 0) {
            opts.onWait?.(`Free-tier quota busy, retrying in ${Math.ceil(d / 1000)}s…`);
            await sleep(d, opts.signal);
            continue;
          }
          break; // long or daily quota: next model
        }
        if (err.status >= 500 || err.status === 404) break; // overloaded / unavailable: next model
        throw err; // 400/401/403: our bug or bad key, don't mask it
      }
    }
    if (i + 1 < GEMINI_MODELS.length) opts.onSwitch?.(model, GEMINI_MODELS[i + 1]);
  }
  throw lastErr instanceof ApiError ? new AllModelsBusyError(lastErr.message) : lastErr;
}

/** zod's JSON Schema output → shape Gemini accepts. */
export function toGeminiSchema(schema: unknown): unknown {
  const s = structuredClone(schema) as Record<string, unknown>;
  delete s.$schema;
  return s;
}

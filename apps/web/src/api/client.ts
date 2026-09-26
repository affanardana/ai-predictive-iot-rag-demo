/**
 * The one place a request is made.
 *
 * Types come from `schema.d.ts`, generated from the API's OpenAPI document by
 * `npm run gen:api`. Nothing here is hand-written against the API's shapes --
 * that is the point of generating, and it is why a schema change shows up as a
 * compile error rather than as an empty table in production.
 */

import { API_BASE_URL } from '@/config'
import { toApiError } from '@/api/errors'

/** A path on this API. Written out so a typo is a compile error, not a 404. */
export type ApiPath =
  | '/api/v1/machines'
  | `/api/v1/machines/${string}`
  | `/api/v1/machines/${string}/telemetry`
  | `/api/v1/machines/${string}/predictions`
  | `/api/v1/machines/${string}/incidents`
  | '/api/v1/incidents'
  | `/api/v1/incidents/${string}`
  | '/api/v1/simulations'
  | `/api/v1/simulations/${string}`
  | `/api/v1/simulations/${string}/stop`

type QueryValue = string | number | boolean | undefined

function buildUrl(path: ApiPath, query?: Record<string, QueryValue>): string {
  const url = new URL(`${API_BASE_URL}${path}`)
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== undefined) url.searchParams.set(key, String(value))
  }
  return url.toString()
}

async function request<T>(
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE',
  path: ApiPath,
  options: { query?: Record<string, QueryValue>; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const response = await fetch(buildUrl(path, options.query), {
    method,
    signal: options.signal,
    headers: options.body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  })

  if (!response.ok) throw await toApiError(response)
  // 204 has no body, and `response.json()` on an empty body throws a parse
  // error that reads like a server fault. `DELETE` is the only route that
  // answers without one.
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

/**
 * Read a resource.
 *
 * `signal` is threaded through because TanStack Query cancels a request whose
 * component has unmounted or whose key changed -- without it, a fast navigation
 * between machines leaves the slower response to arrive last and render the
 * wrong one.
 */
export function getJson<T>(
  path: ApiPath,
  query?: Record<string, QueryValue>,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>('GET', path, { query, signal })
}

/** Send a partial update. */
export function patchJson<T>(path: ApiPath, body: unknown): Promise<T> {
  return request<T>('PATCH', path, { body })
}

/** Create a resource. */
export function postJson<T>(path: ApiPath, body: unknown): Promise<T> {
  return request<T>('POST', path, { body })
}

/** Create a resource with no request body, e.g. an action on an existing one. */
export function postEmpty<T>(path: ApiPath): Promise<T> {
  return request<T>('POST', path)
}

/** Remove a resource. */
export function deleteJson<T = void>(path: ApiPath): Promise<T> {
  return request<T>('DELETE', path)
}

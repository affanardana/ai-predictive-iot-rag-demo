/**
 * The API's error envelope, as a thrown value.
 *
 * Every error the API produces has the same shape -- `{"error": {"code",
 * "message", "details"}}` -- because FastAPI's default `{"detail": ...}` is
 * overridden for every error class. Hand-written rather than generated: the
 * envelope is not in the OpenAPI document, so `schema.d.ts` cannot describe it.
 *
 * `code` is the stable identifier to branch on. `message` is for a human and
 * may change between releases without notice, so nothing here switches on it.
 */

export interface ApiErrorBody {
  error: {
    code: string
    message: string
    details: Record<string, string> | null
  }
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details: Record<string, string> | null = null,
  ) {
    super(message)
    this.name = 'ApiError'
  }

  /**
   * Whether this is a conflict caused by the resource moving under us.
   *
   * The incident actions care: acknowledging an incident somebody else already
   * acknowledged is not a failure to report, it is a cue to refetch. The API
   * answers `409 invalid_incident_transition` for exactly that.
   */
  get isConflict(): boolean {
    return this.status === 409
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== 'object' || value === null || !('error' in value)) return false
  const { error } = value as { error: unknown }
  return typeof error === 'object' && error !== null && 'code' in error && 'message' in error
}

/**
 * Turn a failed response into an `ApiError`.
 *
 * Falls back to a status-derived message when the body is not the expected
 * envelope -- a proxy returning an HTML error page, or a network layer failing
 * before the API is reached. Showing the server's own message when there is one
 * matters: "machine_not_found" is actionable where "Request failed" is not.
 */
export async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown
  try {
    body = await response.json()
  } catch {
    return new ApiError(response.status, 'unreadable_response', describeStatus(response.status))
  }

  if (!isApiErrorBody(body)) {
    return new ApiError(response.status, 'unexpected_error_shape', describeStatus(response.status))
  }

  const { code, message, details } = body.error
  return new ApiError(response.status, code, message, details)
}

function describeStatus(status: number): string {
  if (status === 503) return 'The service is temporarily unavailable.'
  if (status >= 500) return 'The server encountered an error.'
  return `The request failed with status ${status}.`
}

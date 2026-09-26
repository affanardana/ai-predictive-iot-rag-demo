/**
 * Build-time configuration.
 *
 * Read once, at module load, so a missing value fails immediately and in one
 * place. `import.meta.env` is replaced textually by Vite at build time, which
 * is why this cannot be looked up lazily and why the value is baked into the
 * bundle rather than read from the server at runtime.
 */

const DEFAULT_API_BASE_URL = 'http://localhost:8000'

function readApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL
  if (configured === undefined || configured === '') {
    // Not fatal: local development is the overwhelmingly likely reason it is
    // unset, and the default is where `uvicorn api.main:app` listens. A
    // deployed build without it would silently talk to the developer's own
    // machine, so the deployment sets it -- see `.env.example`.
    return DEFAULT_API_BASE_URL
  }
  // A trailing slash would produce `//api/v1/machines`, which some proxies
  // treat as a different path rather than an equivalent one.
  return configured.replace(/\/+$/, '')
}

export const API_BASE_URL = readApiBaseUrl()

/**
 * Where the API's own documentation lives.
 *
 * Surfaced in the UI because this is a portfolio project and a reviewer who
 * wants to check a number against its source should not have to guess the
 * path.
 */
export const API_DOCS_URL = `${API_BASE_URL}/docs`

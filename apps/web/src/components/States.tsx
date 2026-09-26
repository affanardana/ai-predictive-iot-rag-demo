/**
 * What a panel shows when it has nothing, or when something went wrong.
 *
 * The error case is worth its own component because the API's error envelope
 * carries a stable `code` and a human `message`, and showing the server's own
 * words is the difference between "machine_not_found" -- which tells you what
 * to do -- and "Request failed".
 */

import type { ReactNode } from 'react'

import { ApiError } from '@/api/errors'

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return <p className="py-6 text-center text-sm text-ink-muted">{label}</p>
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-ink-muted">{children}</p>
}

export function ErrorState({ error }: { error: unknown }) {
  const message =
    error instanceof ApiError
      ? error.message
      : error instanceof Error
        ? error.message
        : 'Something went wrong.'

  const code = error instanceof ApiError ? error.code : undefined

  return (
    <div className="rounded border border-risk-critical/30 bg-risk-critical/5 px-4 py-3">
      <p className="text-sm font-medium text-risk-critical">{message}</p>
      {code !== undefined && (
        <p className="mt-1 font-mono text-xs text-ink-muted">{code}</p>
      )}
    </div>
  )
}

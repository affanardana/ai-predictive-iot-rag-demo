/**
 * Whether a machine is currently reporting, and how fresh its data is.
 *
 * `MachineSummarySchema` carries `is_reporting` precisely so that "a machine
 * that has stopped reporting keeps its last known values so staleness is
 * visible rather than presenting as health". That intent only holds if the UI
 * actually makes the staleness visible -- otherwise a dead machine with
 * plausible-looking numbers reads as a healthy one, which is the single most
 * dangerous thing this dashboard could do.
 */

import { describeAge } from '@/lib/time'

const TONE: Record<string, string> = {
  live: 'text-risk-normal',
  recent: 'text-ink-muted',
  stale: 'text-risk-high',
  silent: 'text-risk-critical',
  ahead: 'text-accent',
}

export function StaleIndicator({
  recordedAt,
  isReporting,
  now,
}: {
  recordedAt: string | null
  isReporting: boolean
  /** Injectable so the rendering is testable without pinning the clock. */
  now?: Date
}) {
  const age = describeAge(isReporting ? recordedAt : null, now)

  return (
    <span className={`inline-flex items-center gap-1.5 text-sm ${TONE[age.staleness] ?? ''}`}>
      <span aria-hidden="true">{age.staleness === 'live' ? '◉' : '◌'}</span>
      <span>{age.text}</span>
    </span>
  )
}

/**
 * Timestamps, and how old they are.
 */

/** Absolute instant, in the viewer's own timezone. */
export function formatInstant(isoTimestamp: string): string {
  const instant = new Date(isoTimestamp)
  if (Number.isNaN(instant.getTime())) return 'unknown'
  return instant.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** Clock time only, for axis labels where the date is already established. */
export function formatClock(isoTimestamp: string): string {
  const instant = new Date(isoTimestamp)
  if (Number.isNaN(instant.getTime())) return ''
  return instant.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

export type Staleness = 'live' | 'recent' | 'stale' | 'silent' | 'ahead'

export interface AgeDescription {
  staleness: Staleness
  text: string
}

/**
 * How old a reading is, in words.
 *
 * `ahead` is the case worth explaining. The simulator advances its timestamps
 * at `sample_interval x index` while its loop sleeps a fixed tick, so played
 * back quickly its data is genuinely in the future relative to any wall clock
 * -- including the viewer's. The API's telemetry windows follow the data rather
 * than the clock for exactly this reason, and a naive "3 minutes ago" rendered
 * from a negative interval would read as nonsense or, worse, as a machine that
 * had stopped.
 *
 * `now` is a parameter rather than read inside, so the behaviour is testable
 * without pinning the system clock.
 */
export function describeAge(isoTimestamp: string | null, now: Date = new Date()): AgeDescription {
  if (isoTimestamp === null) {
    return { staleness: 'silent', text: 'Never reported' }
  }

  const instant = new Date(isoTimestamp)
  if (Number.isNaN(instant.getTime())) {
    return { staleness: 'silent', text: 'Unknown' }
  }

  const seconds = (now.getTime() - instant.getTime()) / 1000

  if (seconds < -5) {
    // Allow a few seconds of ordinary clock skew between server and browser
    // before calling it "ahead" -- otherwise every machine looks anomalous.
    return { staleness: 'ahead', text: 'Reporting (ahead of your clock)' }
  }
  if (seconds < 90) return { staleness: 'live', text: 'Live' }
  if (seconds < 3600) return { staleness: 'recent', text: `${Math.round(seconds / 60)} min ago` }
  if (seconds < 86_400) return { staleness: 'stale', text: `${Math.round(seconds / 3600)} h ago` }
  return { staleness: 'stale', text: `${Math.round(seconds / 86_400)} d ago` }
}

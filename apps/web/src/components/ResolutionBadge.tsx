/**
 * Whether a chart is showing measurements or averages.
 *
 * `SeriesResolutionSchema` exists so "a consumer must be able to tell a
 * measurement from an aggregate" -- the distinction the PRD's evidence model
 * turns on, and one that is invisible in a line chart. The API buckets anything
 * longer than five hours, so a 24-hour view is two-minute means and a 30-day
 * view is hourly ones, all of which draw as equally authoritative lines unless
 * the UI says otherwise.
 */

import type { SeriesResolution } from '@/api/types'

function describe(resolution: SeriesResolution): string {
  if (resolution.is_raw) return 'Raw measurements'
  const seconds = resolution.bucket_seconds ?? 0
  const width = seconds % 3600 === 0 ? `${seconds / 3600} h` : `${seconds / 60} min`
  return `Aggregated · ${width} means`
}

export function ResolutionBadge({ resolution }: { resolution: SeriesResolution }) {
  const aggregated = !resolution.is_raw

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-medium ${
        aggregated
          ? 'bg-risk-warning/10 text-risk-warning'
          : 'bg-ink-muted/10 text-ink-muted'
      }`}
    >
      <span aria-hidden="true">{aggregated ? '≈' : '·'}</span>
      <span>{describe(resolution)}</span>
    </span>
  )
}

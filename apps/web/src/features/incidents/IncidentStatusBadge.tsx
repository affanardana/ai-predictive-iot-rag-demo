/**
 * An incident's lifecycle state.
 *
 * `ACKNOWLEDGED` is shown as still-open, because that is what the domain says:
 * `Incident.is_open` covers OPEN and ACKNOWLEDGED alike, and the fleet's
 * `open_incident_count` counts both. A UI that painted acknowledged incidents
 * as finished would disagree with the number on the overview page.
 */

import type { IncidentStatus } from '@/api/types'

const PRESENTATION: Record<IncidentStatus, { label: string; className: string; glyph: string }> = {
  OPEN: { label: 'Open', className: 'bg-risk-critical/10 text-risk-critical', glyph: '◆' },
  ACKNOWLEDGED: { label: 'Acknowledged', className: 'bg-risk-warning/10 text-risk-warning', glyph: '▲' },
  RESOLVED: { label: 'Resolved', className: 'bg-risk-normal/10 text-risk-normal', glyph: '✓' },
  DISMISSED: { label: 'Dismissed', className: 'bg-ink-muted/10 text-ink-muted', glyph: '—' },
}

export function IncidentStatusBadge({ status }: { status: IncidentStatus }) {
  const { label, className, glyph } = PRESENTATION[status]

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${className}`}
    >
      <span aria-hidden="true">{glyph}</span>
      <span>{label}</span>
    </span>
  )
}

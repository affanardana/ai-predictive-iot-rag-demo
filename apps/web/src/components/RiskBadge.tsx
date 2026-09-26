/**
 * A risk band, shown three ways at once.
 *
 * Glyph, name, and colour -- never colour alone. The four bands are not
 * reliably separable by hue for every viewer, and a monitoring dashboard whose
 * meaning depends on that is one that misinforms the people most likely to be
 * using it. The shape carries the band in greyscale; the name carries it in
 * speech and to a screen reader.
 */

import type { RiskLevel } from '@/api/types'
import { riskColorVar, riskGlyph, riskLabel } from '@/lib/risk'

export function RiskBadge({ level }: { level: RiskLevel | null }) {
  if (level === null) {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-ink-muted" title="No prediction yet">
        <span aria-hidden="true">○</span>
        <span>No prediction</span>
      </span>
    )
  }

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-sm font-medium"
      style={{
        color: riskColorVar(level),
        backgroundColor: `color-mix(in srgb, ${riskColorVar(level)} 10%, transparent)`,
      }}
    >
      <span aria-hidden="true">{riskGlyph(level)}</span>
      <span>{riskLabel(level)}</span>
    </span>
  )
}

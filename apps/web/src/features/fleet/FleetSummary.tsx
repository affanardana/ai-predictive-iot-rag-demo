/**
 * The counts the overview must communicate, per Journey A in the PRD:
 * total machines, machines in normal state, machines requiring attention,
 * critical machines, and active incidents.
 *
 * The bands are distinguished by name and shape as well as by colour, for the
 * same reason `RiskBadge` is.
 */

import type { MachineSummary, RiskLevel } from '@/api/types'
import { needsAttention, RISK_ORDER, riskColorVar, riskGlyph, riskLabel } from '@/lib/risk'

export function FleetSummary({ summaries }: { summaries: readonly MachineSummary[] }) {
  const byBand = new Map<RiskLevel, number>(RISK_ORDER.map((level) => [level, 0]))
  let noPrediction = 0
  let openIncidents = 0
  let needingAttention = 0

  for (const summary of summaries) {
    if (summary.risk_level === null) noPrediction += 1
    else byBand.set(summary.risk_level, (byBand.get(summary.risk_level) ?? 0) + 1)

    openIncidents += summary.open_incident_count
    if (needsAttention(summary.risk_level)) needingAttention += 1
  }

  // Exactly one hero figure, because a page of equally large numbers has no
  // hierarchy and the reader has to work out what matters.
  return (
    <div className="space-y-4">
      <div>
        <p className="text-sm text-ink-muted">Machines at High or Critical</p>
        <p className="text-4xl font-semibold tracking-tight">
          {needingAttention}
          <span className="ml-2 text-lg font-normal text-ink-muted">of {summaries.length}</span>
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {RISK_ORDER.map((level) => (
          <Tile
            key={level}
            label={riskLabel(level)}
            glyph={riskGlyph(level)}
            colour={riskColorVar(level)}
            value={byBand.get(level) ?? 0}
          />
        ))}
        <Tile label="No prediction" glyph="○" colour="#5b6472" value={noPrediction} />
        <Tile
          label="Open incidents"
          glyph="⚠"
          colour={openIncidents > 0 ? 'var(--color-risk-critical)' : '#5b6472'}
          value={openIncidents}
        />
      </div>
    </div>
  )
}

function Tile({
  label,
  glyph,
  colour,
  value,
}: {
  label: string
  glyph: string
  colour: string
  value: number
}) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2">
      <div className="flex items-center gap-1.5 text-xs font-medium" style={{ color: colour }}>
        <span aria-hidden="true">{glyph}</span>
        <span>{label}</span>
      </div>
      <p className="mt-1 text-xl font-semibold">{value}</p>
    </div>
  )
}

/**
 * The fleet, as a table.
 *
 * Columns are the ones PRD section 7 requires a machine to expose, plus the
 * staleness indicator -- which is not decoration. A machine that has stopped
 * reporting keeps its last known values, so without it a dead machine renders
 * as a healthy one with plausible numbers. That is the most dangerous thing
 * this dashboard could show.
 */

import { Link } from 'react-router-dom'

import { RiskBadge } from '@/components/RiskBadge'
import { StaleIndicator } from '@/components/StaleIndicator'
import type { MachineSummary, SensorSignal } from '@/api/types'
import { formatProbability } from '@/lib/risk'
import { formatClock } from '@/lib/time'

/** One decimal: these are compared down a column, not read as precise values. */
function reading(summary: MachineSummary, signal: SensorSignal): string {
  const reading = summary.latest_telemetry?.reading
  if (reading === undefined) return '—'
  return reading[signal].toFixed(1)
}

export function MachineTable({ summaries }: { summaries: readonly MachineSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[52rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-ink-muted">
            <th className="py-2 pr-4 font-medium">Machine</th>
            <th className="py-2 pr-4 font-medium">Risk</th>
            <th className="py-2 pr-4 text-right font-medium">Temp °C</th>
            <th className="py-2 pr-4 text-right font-medium">Vib mm/s</th>
            <th className="py-2 pr-4 text-right font-medium">RPM</th>
            <th className="py-2 pr-4 text-right font-medium">Current A</th>
            <th className="py-2 pr-4 text-right font-medium">Failure</th>
            <th className="py-2 pr-4 text-right font-medium">Incidents</th>
            <th className="py-2 font-medium">Last update</th>
          </tr>
        </thead>
        <tbody>
          {summaries.map((summary) => {
            const machineId = summary.machine.machine_id
            return (
              <tr key={machineId} className="border-b border-line last:border-0 hover:bg-canvas">
                <td className="py-3 pr-4">
                  <Link to={`/machines/${machineId}`} className="font-medium hover:text-accent">
                    {machineId}
                  </Link>
                  <div className="text-xs text-ink-muted">{summary.machine.name}</div>
                </td>
                <td className="py-3 pr-4">
                  <RiskBadge level={summary.risk_level} />
                </td>
                <td className="py-3 pr-4 text-right">{reading(summary, 'temperature')}</td>
                <td className="py-3 pr-4 text-right">{reading(summary, 'vibration')}</td>
                <td className="py-3 pr-4 text-right">{reading(summary, 'rpm')}</td>
                <td className="py-3 pr-4 text-right">{reading(summary, 'current')}</td>
                <td className="py-3 pr-4 text-right">
                  {summary.latest_prediction === null
                    ? '—'
                    : formatProbability(summary.latest_prediction.failure_probability)}
                </td>
                <td className="py-3 pr-4 text-right">
                  {summary.open_incident_count > 0 ? (
                    <span className="font-medium text-risk-high">{summary.open_incident_count}</span>
                  ) : (
                    <span className="text-ink-muted">0</span>
                  )}
                </td>
                <td className="py-3">
                  <StaleIndicator
                    recordedAt={summary.latest_telemetry?.recorded_at ?? null}
                    isReporting={summary.is_reporting}
                  />
                  {summary.latest_telemetry !== null && (
                    <div className="text-xs text-ink-muted">
                      {formatClock(summary.latest_telemetry.recorded_at)}
                    </div>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

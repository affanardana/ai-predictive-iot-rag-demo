/**
 * Incidents, as a table.
 *
 * Carries every field PRD section 10 requires an incident to have, and the
 * actions that make the lifecycle reachable. Note what `incident_type` shows:
 * `UNCLASSIFIED`, always, because the model is a binary classifier and cannot
 * say what is failing. That is a known and recorded gap, left visible in the
 * product rather than hidden, and the column says so in as many words.
 */

import { Link } from 'react-router-dom'

import type { Incident } from '@/api/types'
import { IncidentActions } from '@/features/incidents/IncidentActions'
import { IncidentStatusBadge } from '@/features/incidents/IncidentStatusBadge'
import { formatProbability } from '@/lib/risk'
import { formatInstant } from '@/lib/time'

export function IncidentTable({
  incidents,
  showMachine = true,
}: {
  incidents: readonly Incident[]
  showMachine?: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[48rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-ink-muted">
            {showMachine && <th className="py-2 pr-4 font-medium">Machine</th>}
            <th className="py-2 pr-4 font-medium">Failure mode</th>
            <th className="py-2 pr-4 font-medium">Severity</th>
            <th className="py-2 pr-4 text-right font-medium">Probability</th>
            <th className="py-2 pr-4 font-medium">Detected</th>
            <th className="py-2 pr-4 font-medium">Status</th>
            <th className="py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {incidents.map((incident) => (
            <tr key={incident.incident_id} className="border-b border-line last:border-0">
              {showMachine && (
                <td className="py-3 pr-4">
                  <Link
                    to={`/machines/${incident.machine_id}`}
                    className="font-medium hover:text-accent"
                  >
                    {incident.machine_id}
                  </Link>
                </td>
              )}
              <td className="py-3 pr-4">
                <span className="text-ink-muted" title="The model detects rising risk, not its cause.">
                  Unclassified
                </span>
              </td>
              <td className="py-3 pr-4">{incident.severity}</td>
              <td className="py-3 pr-4 text-right">
                {formatProbability(incident.failure_probability)}
              </td>
              <td className="py-3 pr-4 whitespace-nowrap">{formatInstant(incident.detected_at)}</td>
              <td className="py-3 pr-4">
                <IncidentStatusBadge status={incident.status} />
              </td>
              <td className="py-3">
                <IncidentActions incident={incident} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

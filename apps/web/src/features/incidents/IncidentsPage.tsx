/**
 * The incident list.
 *
 * "One incident per excursion" is the shape to expect, not one per reading.
 * `RecordPrediction` raises an incident only when a machine has none open, so a
 * machine that crosses into High and stays there files one incident rather than
 * hundreds. That rule is a Phase 6 invention, not something the PRD states, and
 * it is the reason this list is readable at all.
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { incidentListQuery } from '@/api/queries'
import type { IncidentStatus } from '@/api/types'
import { Card } from '@/components/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { IncidentTable } from '@/features/incidents/IncidentTable'

const FILTERS: readonly { label: string; status: IncidentStatus | undefined }[] = [
  { label: 'All', status: undefined },
  { label: 'Open', status: 'OPEN' },
  { label: 'Acknowledged', status: 'ACKNOWLEDGED' },
  { label: 'Resolved', status: 'RESOLVED' },
  { label: 'Dismissed', status: 'DISMISSED' },
]

export function IncidentsPage() {
  const [status, setStatus] = useState<IncidentStatus | undefined>(undefined)
  const incidents = useQuery(incidentListQuery(status))

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Incidents</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Raised when a machine's predicted band reaches High or Critical, and suppressed while one
          is already open for that machine.
        </p>
      </div>

      <Card
        actions={
          <div className="flex rounded border border-line">
            {FILTERS.map((filter) => (
              <button
                key={filter.label}
                type="button"
                onClick={() => setStatus(filter.status)}
                className={`px-2.5 py-1 text-xs ${
                  filter.status === status
                    ? 'bg-accent/10 font-medium text-accent'
                    : 'text-ink-muted hover:bg-canvas'
                }`}
              >
                {filter.label}
              </button>
            ))}
          </div>
        }
      >
        {incidents.isPending && <LoadingState label="Loading incidents…" />}
        {incidents.isError && <ErrorState error={incidents.error} />}
        {incidents.data !== undefined &&
          (incidents.data.length === 0 ? (
            <EmptyState>
              No incidents{status === undefined ? '' : ' with this status'}. A healthy fleet
              produces none, and a degrading one produces exactly one per excursion.
            </EmptyState>
          ) : (
            <IncidentTable incidents={incidents.data} />
          ))}
      </Card>
    </div>
  )
}

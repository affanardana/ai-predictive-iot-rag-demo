/**
 * The buttons that move an incident through its lifecycle.
 *
 * Which buttons appear is decided by the incident's current status, mirroring
 * the transition table in `api/domain/value_objects/incident_status.py`. That
 * table is the authority; this only avoids offering a move the server would
 * refuse, and the refusal is handled anyway -- see `onConflict`.
 */

import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'

import { keys, updateIncidentStatus } from '@/api/queries'
import { ApiError } from '@/api/errors'
import type { Incident, IncidentStatus } from '@/api/types'

/** The moves the API permits from each status. */
const ALLOWED: Record<IncidentStatus, readonly { status: IncidentStatus; label: string }[]> = {
  OPEN: [
    { status: 'ACKNOWLEDGED', label: 'Acknowledge' },
    { status: 'RESOLVED', label: 'Resolve' },
    { status: 'DISMISSED', label: 'Dismiss' },
  ],
  ACKNOWLEDGED: [
    { status: 'RESOLVED', label: 'Resolve' },
    { status: 'DISMISSED', label: 'Dismiss' },
  ],
  // Terminal. Reopening is modelled as a new incident so the audit trail stays
  // append-only -- which is why there is nothing to offer here.
  RESOLVED: [],
  DISMISSED: [],
}

export function IncidentActions({ incident }: { incident: Incident }) {
  const queryClient = useQueryClient()
  const [notice, setNotice] = useState<string | undefined>(undefined)

  const mutation = useMutation({
    mutationFn: (status: IncidentStatus) => updateIncidentStatus(incident.incident_id, status),

    // Optimistic, so the row changes the instant it is clicked. The invalidation
    // below replaces this with the server's answer a moment later.
    onMutate: async (status) => {
      setNotice(undefined)
      await queryClient.cancelQueries({ queryKey: keys.incidents })
      queryClient.setQueriesData<Incident[]>({ queryKey: keys.incidents }, (previous) =>
        previous?.map((item) =>
          item.incident_id === incident.incident_id ? { ...item, status } : item,
        ),
      )
    },

    onError: (error) => {
      // A 409 is not a failure worth alarming anyone about. The table has no
      // self-transitions, so it means somebody else acted on this incident
      // between the page loading and the click -- a second tab, or a colleague.
      // The right response is to show the current truth, not to report a fault.
      if (error instanceof ApiError && error.isConflict) {
        setNotice('This incident was already updated elsewhere. Showing the current state.')
      } else {
        setNotice(error instanceof Error ? error.message : 'That change did not go through.')
      }
    },

    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: keys.incidents })
      void queryClient.invalidateQueries({ queryKey: keys.all })
      void queryClient.invalidateQueries({ queryKey: keys.machine(incident.machine_id) })
    },
  })

  const moves = ALLOWED[incident.status]

  if (moves.length === 0 && notice === undefined) {
    return <span className="text-xs text-ink-muted">Closed</span>
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {moves.map((move) => (
        <button
          key={move.status}
          type="button"
          // Disabled while a move is in flight, which is what stops a
          // double-click from becoming a 409 the user then has to interpret.
          disabled={mutation.isPending}
          onClick={() => mutation.mutate(move.status)}
          className="rounded border border-line px-2 py-1 text-xs hover:bg-canvas disabled:cursor-not-allowed disabled:opacity-50"
        >
          {move.label}
        </button>
      ))}
      {notice !== undefined && <span className="text-xs text-ink-muted">{notice}</span>}
    </div>
  )
}

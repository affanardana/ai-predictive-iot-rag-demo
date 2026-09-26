/**
 * Which cached queries a change invalidates.
 *
 * A pure function, separate from the provider that calls it, because this is
 * the contract between the server's event vocabulary and the UI's freshness --
 * and it is the kind of contract that fails silently. A key that stops
 * matching produces a dashboard that renders stale numbers with no error
 * anywhere, so it is worth a unit test rather than a `useEffect`.
 */

import { keys } from '@/api/queries'
import type { MachineEvent } from '@/realtime/types'

/** A query key prefix, as TanStack Query matches it. */
export type QueryKey = readonly unknown[]

export function invalidationsFor(event: MachineEvent): readonly QueryKey[] {
  switch (event.kind) {
    case 'TELEMETRY_RECEIVED':
      // The newest reading is on the fleet row and in every open telemetry
      // window for that machine. The detail response carries the latest
      // reading too, so it moves as well.
      return [keys.fleet(), keys.machineDetail(event.machine_id), keys.telemetryAll(event.machine_id)]

    case 'PREDICTION_RECORDED':
      // A scoring moves the probability on the fleet row, the machine's own
      // state, its history, and the incident list -- because a scoring that
      // crosses a band raises an incident alongside it.
      return [
        keys.fleet(),
        keys.machineDetail(event.machine_id),
        keys.predictions(event.machine_id),
        keys.machineIncidents(event.machine_id),
        keys.incidents,
      ]

    case 'INCIDENT_RAISED':
      return [
        keys.fleet(),
        keys.machineDetail(event.machine_id),
        keys.machineIncidents(event.machine_id),
        keys.incidents,
      ]

    case 'INCIDENT_STATUS_CHANGED':
      // Announced against the machine rather than the incident, because
      // resolving one changes `open_incident_count` -- a number on the fleet
      // page, not only on the incident page.
      return [
        keys.fleet(),
        keys.machineDetail(event.machine_id),
        keys.machineIncidents(event.machine_id),
        keys.incidents,
      ]

    case 'SIMULATION_STATE_CHANGED':
      // The runs list and the machine's own panel. Deliberately **not** the
      // fleet: a run starting or stopping changes no fleet number. The readings
      // that follow arrive as `TELEMETRY_RECEIVED` and move those themselves,
      // so invalidating here would be a refetch for nothing.
      return [keys.simulations(), keys.machineSimulations(event.machine_id)]
  }
}

/**
 * Everything, for a reconnect.
 *
 * `EventSource` reconnects on its own, and frames sent while it was
 * disconnected are gone. Because the payload is a hint rather than the data,
 * the recovery for a gap and the recovery for a fresh connection are the same
 * action -- which is what removes the need for a replay buffer, sequence
 * tracking, and everything that would come with them.
 */
export function everything(): readonly QueryKey[] {
  return [keys.all, keys.incidents]
}

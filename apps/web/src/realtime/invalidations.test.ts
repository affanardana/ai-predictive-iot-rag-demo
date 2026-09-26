import { describe, expect, it } from 'vitest'

import { keys } from '@/api/queries'
import { invalidationsFor } from '@/realtime/invalidations'
import { EVENT_KINDS, type EventKind } from '@/realtime/types'

const MACHINE = 'M003'

function keysFor(kind: EventKind): string[] {
  return invalidationsFor({ kind, machine_id: MACHINE }).map((key) => JSON.stringify(key))
}

describe('invalidationsFor', () => {
  it('handles every event kind the server can send', () => {
    // The switch is exhaustive over `EventKind`, so this fails at typecheck
    // time if a kind is added and at runtime if one is ever filtered out. The
    // failure mode it prevents is a dashboard that renders stale numbers with
    // no error anywhere -- which is why it is asserted rather than assumed.
    for (const kind of EVENT_KINDS) {
      expect(keysFor(kind).length).toBeGreaterThan(0)
    }
  })

  it('refreshes the fleet and the machine when telemetry arrives', () => {
    const invalidated = keysFor('TELEMETRY_RECEIVED')

    expect(invalidated).toContain(JSON.stringify(keys.fleet()))
    expect(invalidated).toContain(JSON.stringify(keys.telemetryAll(MACHINE)))
  })

  it('refreshes prediction history and incidents when a machine is scored', () => {
    // A scoring that crosses a band raises an incident alongside it, so the
    // incident views have to move even though the event names a prediction.
    const invalidated = keysFor('PREDICTION_RECORDED')

    expect(invalidated).toContain(JSON.stringify(keys.predictions(MACHINE)))
    expect(invalidated).toContain(JSON.stringify(keys.incidents))
  })

  it('refreshes the fleet when an incident status changes', () => {
    // The subtle one. A status change is announced against the machine, not
    // the incident, because resolving an incident drops it out of
    // `open_incident_count` -- a number on the fleet page.
    const invalidated = keysFor('INCIDENT_STATUS_CHANGED')

    expect(invalidated).toContain(JSON.stringify(keys.fleet()))
    expect(invalidated).toContain(JSON.stringify(keys.machineIncidents(MACHINE)))
  })

  it('refreshes the run lists when a simulation changes state', () => {
    const invalidated = keysFor('SIMULATION_STATE_CHANGED')

    expect(invalidated).toContain(JSON.stringify(keys.simulations()))
    expect(invalidated).toContain(JSON.stringify(keys.machineSimulations(MACHINE)))
  })

  it('does not refresh the fleet when a simulation changes state', () => {
    // A run starting or stopping changes no fleet number. The readings that
    // follow arrive as TELEMETRY_RECEIVED and move those themselves, so
    // invalidating here would be a refetch for nothing -- on the endpoint a
    // one-core box can least afford it.
    const invalidated = keysFor('SIMULATION_STATE_CHANGED')

    expect(invalidated).not.toContain(JSON.stringify(keys.fleet()))
  })

  it('never invalidates another machine', () => {
    // Prefix matching means a key scoped to one machine can accidentally match
    // the fleet, but not the reverse. If this ever changes, every event would
    // refetch every machine.
    for (const kind of EVENT_KINDS) {
      const others = invalidationsFor({ kind, machine_id: MACHINE }).filter((key) =>
        key.includes('M004'),
      )
      expect(others).toEqual([])
    }
  })
})

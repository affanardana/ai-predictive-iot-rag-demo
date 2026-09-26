/**
 * The event stream's payload, mirrored by hand.
 *
 * **This is the one place the "never hand-write the client" rule is broken**,
 * and it is broken for a reason rather than by oversight: a
 * `text/event-stream` body cannot be described in OpenAPI, so it cannot be
 * generated from the document. `apps/api/tests/integration/test_events_api.py`
 * pins both the field names and every `EventKind` value, so a rename on the
 * server fails the Python suite rather than silently freezing this app.
 *
 * Fields are snake_case because they are the server's, unmodified. Translating
 * them here would add a layer that exists only to be got wrong.
 */

/**
 * What changed. One member per event the server emits.
 *
 * Ordering the array and deriving the union from it means the exhaustiveness
 * check in `invalidationsFor` is total: adding a fifth kind on the server makes
 * `npm run typecheck` fail here until it is handled, which is the behaviour
 * wanted from a contract this one cannot be generated from.
 */
export const EVENT_KINDS = [
  'TELEMETRY_RECEIVED',
  'PREDICTION_RECORDED',
  'INCIDENT_RAISED',
  'INCIDENT_STATUS_CHANGED',
  'SIMULATION_STATE_CHANGED',
] as const

export type EventKind = (typeof EVENT_KINDS)[number]

export interface MachineEvent {
  kind: EventKind
  machine_id: string
}

function isEventKind(value: unknown): value is EventKind {
  return typeof value === 'string' && (EVENT_KINDS as readonly string[]).includes(value)
}

/**
 * Parse a frame's payload, or return undefined.
 *
 * Validating rather than casting because the bytes arrive from the network
 * unbidden, and a malformed frame that reached a `switch` unchecked would fall
 * through every case silently. Returning `undefined` lets the caller ignore it,
 * which is the right response to one bad frame out of a stream: the next event
 * repairs whatever it was about to say.
 */
export function parseMachineEvent(payload: string): MachineEvent | undefined {
  let decoded: unknown
  try {
    decoded = JSON.parse(payload)
  } catch {
    return undefined
  }

  if (typeof decoded !== 'object' || decoded === null) return undefined
  const candidate = decoded as { kind?: unknown; machine_id?: unknown }
  if (!isEventKind(candidate.kind) || typeof candidate.machine_id !== 'string') return undefined

  return { kind: candidate.kind, machine_id: candidate.machine_id }
}

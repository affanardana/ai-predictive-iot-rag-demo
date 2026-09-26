import { describe, expect, it } from 'vitest'

import { EVENT_KINDS, parseMachineEvent } from '@/realtime/types'

describe('parseMachineEvent', () => {
  it('accepts a frame the server sends', () => {
    expect(parseMachineEvent('{"kind":"TELEMETRY_RECEIVED","machine_id":"M003"}')).toEqual({
      kind: 'TELEMETRY_RECEIVED',
      machine_id: 'M003',
    })
  })

  it('accepts every kind the server can send', () => {
    for (const kind of EVENT_KINDS) {
      expect(parseMachineEvent(`{"kind":"${kind}","machine_id":"M003"}`)?.kind).toBe(kind)
    }
  })

  it('rejects a kind it does not know', () => {
    // The forward-compatibility case. A server that starts sending a fifth
    // kind must not make an older page fall through every case in
    // `invalidationsFor` and silently stop updating; dropping the frame keeps
    // the polling floor in charge until the page is redeployed.
    expect(parseMachineEvent('{"kind":"SOMETHING_NEW","machine_id":"M003"}')).toBeUndefined()
  })

  it('rejects a frame that is missing its machine', () => {
    expect(parseMachineEvent('{"kind":"TELEMETRY_RECEIVED"}')).toBeUndefined()
  })

  it('rejects malformed JSON rather than throwing', () => {
    // Frames arrive unbidden. One bad frame out of a stream is not worth an
    // error boundary -- the next event repairs whatever it was about to say.
    expect(parseMachineEvent('not json')).toBeUndefined()
    expect(parseMachineEvent('')).toBeUndefined()
  })

  it('rejects a JSON value that is not an object', () => {
    expect(parseMachineEvent('"TELEMETRY_RECEIVED"')).toBeUndefined()
    expect(parseMachineEvent('null')).toBeUndefined()
  })

  it('carries no telemetry, which is the whole design', () => {
    // If a reading ever appeared in a frame, the stream would have become a
    // second source of truth: the client would be accumulating state from it
    // rather than refetching, and a dropped frame would start to matter.
    const parsed = parseMachineEvent(
      '{"kind":"TELEMETRY_RECEIVED","machine_id":"M003","temperature":900}',
    )

    expect(parsed).toEqual({ kind: 'TELEMETRY_RECEIVED', machine_id: 'M003' })
    expect(Object.keys(parsed ?? {})).toEqual(['kind', 'machine_id'])
  })
})

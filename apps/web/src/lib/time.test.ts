import { describe, expect, it } from 'vitest'

import { describeAge } from '@/lib/time'

const NOW = new Date('2026-09-26T12:00:00Z')

function ago(seconds: number): string {
  return new Date(NOW.getTime() - seconds * 1000).toISOString()
}

describe('describeAge', () => {
  it('calls a fresh reading live', () => {
    expect(describeAge(ago(10), NOW).staleness).toBe('live')
  })

  it('says a machine has never reported when there is no reading', () => {
    expect(describeAge(null, NOW)).toEqual({ staleness: 'silent', text: 'Never reported' })
  })

  it('reports data ahead of the clock as ahead, not as a negative age', () => {
    // The case the clock-skew fix creates. The simulator advances its
    // timestamps faster than real time, so a played-back run genuinely has
    // data in the future relative to the viewer's clock. Rendering "in -3
    // minutes" -- or worse, "3 minutes ago" from a negative interval -- would
    // read as a broken dashboard rather than as a fast-forwarded simulation.
    const { staleness, text } = describeAge(ago(-300), NOW)

    expect(staleness).toBe('ahead')
    expect(text).not.toMatch(/-\d/)
    expect(text).not.toMatch(/\d+ min ago/)
  })

  it('tolerates small clock differences between server and browser', () => {
    // Server and browser clocks disagree by seconds routinely. Calling that
    // "ahead" would mark every machine anomalous on a machine whose clock is
    // slightly fast.
    expect(describeAge(ago(-2), NOW).staleness).toBe('live')
  })

  it('escalates through minutes, hours and days', () => {
    expect(describeAge(ago(600), NOW).text).toBe('10 min ago')
    expect(describeAge(ago(7200), NOW).text).toBe('2 h ago')
    expect(describeAge(ago(172_800), NOW).text).toBe('2 d ago')
  })

  it('marks a machine that stopped reporting hours ago as stale', () => {
    // The safety property. A dead machine keeps its last known values, so the
    // only thing standing between a reader and "this machine is fine" is this
    // classification.
    expect(describeAge(ago(7200), NOW).staleness).toBe('stale')
  })
})

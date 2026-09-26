import { describe, expect, it } from 'vitest'

import { RISK_ORDER, formatProbability, needsAttention, riskGlyph, riskLabel } from '@/lib/risk'

describe('risk bands', () => {
  it('gives every band a distinct glyph', () => {
    // Glyph, not colour, is what makes the bands separable in greyscale and to
    // a reader who cannot distinguish the hues. Two bands sharing a shape
    // would quietly remove that.
    const glyphs = RISK_ORDER.map(riskGlyph)

    expect(new Set(glyphs).size).toBe(RISK_ORDER.length)
  })

  it('gives every band a distinct name', () => {
    expect(new Set(RISK_ORDER.map(riskLabel)).size).toBe(RISK_ORDER.length)
  })

  it('treats only High and Critical as needing attention', () => {
    expect(RISK_ORDER.filter(needsAttention)).toEqual(['HIGH', 'CRITICAL'])
  })

  it('treats a missing band as not needing attention', () => {
    // A machine with no prediction is not a machine in trouble. Counting it
    // would make the overview's headline figure wrong on a fresh deployment.
    expect(needsAttention(null)).toBe(false)
    expect(needsAttention(undefined)).toBe(false)
  })

  it('formats a probability with one decimal', () => {
    // One decimal rather than none: the demonstration's interesting moment is
    // a value climbing through the nineties, and whole percent hides it.
    expect(formatProbability(0.9982)).toBe('99.8%')
    expect(formatProbability(0)).toBe('0.0%')
    expect(formatProbability(1)).toBe('100.0%')
  })
})

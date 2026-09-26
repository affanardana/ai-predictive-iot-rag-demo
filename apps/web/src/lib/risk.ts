/**
 * Risk bands, as the UI presents them.
 *
 * The bands are product configuration rather than model output: the LSTM emits
 * a probability and the API derives the band from thresholds that live in the
 * server's environment. Nothing here recomputes them -- the API sends
 * `risk_level` on every prediction and summary, and this module only decides
 * how to show it.
 */

import type { RiskLevel } from '@/api/types'

/** Ordered least to most severe, so a fleet can be sorted by urgency. */
export const RISK_ORDER: readonly RiskLevel[] = ['NORMAL', 'WARNING', 'HIGH', 'CRITICAL']

export function riskRank(level: RiskLevel): number {
  return RISK_ORDER.indexOf(level)
}

/**
 * The band's colour token.
 *
 * Colour is never the only signal. Every place a band is shown, it is shown
 * with its name and a shape -- see `riskGlyph` -- because roughly one man in
 * twelve cannot reliably tell the warning and critical hues apart, and a
 * dashboard whose meaning depends on that is a dashboard that lies to them.
 */
export function riskColorVar(level: RiskLevel): string {
  return `var(--color-risk-${level.toLowerCase()})`
}

/**
 * A shape that differs per band, so the levels survive greyscale and
 * colour-vision deficiency.
 */
export function riskGlyph(level: RiskLevel): string {
  switch (level) {
    case 'NORMAL':
      return '●'
    case 'WARNING':
      return '▲'
    case 'HIGH':
      return '◆'
    case 'CRITICAL':
      return '■'
  }
}

/**
 * A band's name as prose.
 *
 * `NORMAL` is shouted by the API because it is an enum; a human reading a
 * dashboard should not have to be.
 */
export function riskLabel(level: RiskLevel): string {
  switch (level) {
    case 'NORMAL':
      return 'Normal'
    case 'WARNING':
      return 'Warning'
    case 'HIGH':
      return 'High'
    case 'CRITICAL':
      return 'Critical'
  }
}

/** Whether a band warrants attention, which is what the overview counts. */
export function needsAttention(level: RiskLevel | null | undefined): boolean {
  return level === 'HIGH' || level === 'CRITICAL'
}

/**
 * A probability, as a percentage with one decimal.
 *
 * One decimal rather than none: the demo's interesting moment is a value
 * climbing through 0.9x, and rounding to whole percent hides the movement that
 * the whole project is about.
 */
export function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`
}

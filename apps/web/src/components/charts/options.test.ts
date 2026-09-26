import { describe, expect, it } from 'vitest'

import { SENSOR_SIGNALS } from '@/api/types'
import type { Prediction, TelemetrySeries } from '@/api/types'
import { buildPredictionOption, buildTelemetryOption } from '@/components/charts/options'

const READING = {
  temperature: 65,
  vibration: 1.4,
  rpm: 1480,
  current: 12.5,
  load: 0.72,
  voltage: 400,
}

function series(overrides: Partial<TelemetrySeries> = {}): TelemetrySeries {
  return {
    machine_id: 'M003',
    window: '1h',
    resolution: { is_raw: true, bucket_seconds: null, aggregation: 'RAW' },
    interval: { start: '2026-09-26T11:00:00Z', end: '2026-09-26T12:00:00Z' },
    points: [
      { timestamp: '2026-09-26T11:30:00Z', reading: READING, sample_count: 1 },
      { timestamp: '2026-09-26T12:00:00Z', reading: READING, sample_count: 1 },
    ],
    ...overrides,
  }
}

function predictions(probabilities: [number, Prediction['risk_level']][]): Prediction[] {
  return probabilities.map(([probability, risk_level], index) => ({
    prediction_id: `pred-${index}`,
    machine_id: 'M003',
    predicted_at: new Date(Date.UTC(2026, 8, 26, 12, index)).toISOString(),
    failure_probability: probability,
    risk_level,
    horizon_seconds: 3600,
    model_version: 'lstm-v1',
  }))
}

describe('buildTelemetryOption', () => {
  it('draws one panel per signal rather than one chart with six series', () => {
    // Six signals share no unit. On one axis the relative heights would encode
    // degrees, millimetres per second, revolutions per minute, amperes, a
    // dimensionless load and volts -- a comparison that means nothing.
    const option = buildTelemetryOption(series())
    const drawn = option.series as { xAxisIndex: number; yAxisIndex: number }[]

    expect(drawn).toHaveLength(SENSOR_SIGNALS.length)
    expect(new Set(drawn.map((item) => item.yAxisIndex)).size).toBe(SENSOR_SIGNALS.length)
  })

  it('takes the x-axis extent from the data, never from the clock', () => {
    // The rule the clock-skew fix makes load-bearing. With `max: Date.now()`
    // a fast-forwarded run would sit entirely outside the visible range: the
    // series would exist, the chart would be blank, and it would look like an
    // empty database rather than a rendering choice.
    const option = buildTelemetryOption(series())
    const axes = option.xAxis as { min?: number; max?: number }[]
    const last = new Date('2026-09-26T12:00:00Z').getTime()

    expect(axes[0]?.max).toBe(last)
    expect(axes[0]?.max).not.toBe(Date.now())
  })

  it('leaves the extent undefined when there is nothing to plot', () => {
    const option = buildTelemetryOption(series({ points: [] }))
    const axes = option.xAxis as { min?: number; max?: number }[]

    expect(axes[0]?.min).toBeUndefined()
  })

  it('marks an aggregated series as aggregated', () => {
    // `SeriesResolutionSchema` exists so a consumer can tell a measurement from
    // an average. Dashing the line is how that reaches the screen.
    const option = buildTelemetryOption(
      series({ resolution: { is_raw: false, bucket_seconds: 120, aggregation: 'MEAN' } }),
    )
    const [drawn] = option.series as { lineStyle: { type: string }; showSymbol: boolean }[]

    expect(drawn?.lineStyle.type).toBe('dashed')
    expect(drawn?.showSymbol).toBe(true)
  })

  it('draws a raw series as a solid line with no markers', () => {
    const option = buildTelemetryOption(series())
    const [drawn] = option.series as { lineStyle: { type: string }; showSymbol: boolean }[]

    expect(drawn?.lineStyle.type).toBe('solid')
    expect(drawn?.showSymbol).toBe(false)
  })
})

describe('buildPredictionOption', () => {
  it('orders predictions oldest first', () => {
    const option = buildPredictionOption(predictions([[0.9, 'CRITICAL'], [0.2, 'NORMAL']]))
    const [drawn] = option.series as { data: { value: [number, number] }[] }[]
    const times = drawn?.data.map((point) => point.value[0]) ?? []

    expect(times).toEqual([...times].sort((left, right) => left - right))
  })

  it('colours each point from the band the server assigned', () => {
    // Not recomputed from thresholds here. The bands are the API's product
    // configuration, and a second copy of the boundaries in the frontend is
    // exactly the drift this project avoids elsewhere.
    const option = buildPredictionOption(predictions([[0.95, 'CRITICAL']]))
    const [drawn] = option.series as { data: { itemStyle: { color: string } }[] }[]

    expect(drawn?.data[0]?.itemStyle.color).toBe('var(--color-risk-critical)')
  })

  it('pins the y-axis to the full probability range', () => {
    // A scaled axis would make a rise from 0.02 to 0.06 look like a crisis.
    const option = buildPredictionOption(predictions([[0.04, 'NORMAL']]))
    const axis = option.yAxis as { min: number; max: number }

    expect(axis.min).toBe(0)
    expect(axis.max).toBe(1)
  })
})

/**
 * Chart options, as pure functions.
 *
 * Separated from the components that render them so they can be unit-tested
 * without a DOM, a canvas, or a live API -- which is the only way the rules
 * below are enforced rather than intended.
 */

import type { EChartsOption } from 'echarts'

import { SENSOR_SIGNALS, SIGNAL_UNITS, type Prediction, type SensorSignal, type TelemetrySeries } from '@/api/types'
import { formatProbability, riskColorVar, riskLabel, riskRank } from '@/lib/risk'

/** Shared styling, so the six panels look like one figure. */
const AXIS_LABEL_COLOUR = '#5b6472'
const GRID_LINE_COLOUR = '#e2e5ea'
const LINE_WIDTH = 2

/**
 * The one series colour.
 *
 * Every telemetry panel uses it. In small multiples each panel holds a single
 * series, so there is no identity for colour to carry -- identity is positional
 * and named by the panel's title. Six hues would be colour spent distinguishing
 * things that are already distinguished, and would fight the risk bands for
 * attention.
 */
const SERIES_COLOUR = '#2563eb'

interface Point {
  /** Milliseconds since the epoch, which is what a `time` axis expects. */
  time: number
  value: number
  sampleCount: number
}

function seriesFor(series: TelemetrySeries, signal: SensorSignal): Point[] {
  return series.points.map((point) => ({
    time: new Date(point.timestamp).getTime(),
    value: point.reading[signal],
    sampleCount: point.sample_count,
  }))
}

/**
 * The x-axis extent, taken from the data and never from the wall clock.
 *
 * This is the one chart rule that the clock-skew fix makes load-bearing. A
 * `time` axis with `max: Date.now()` would place a fast-forwarded run entirely
 * outside the visible range -- the series would exist, the chart would be
 * blank, and it would look like an empty database. The API's telemetry windows
 * follow the data for exactly the same reason.
 */
function extent(points: Point[]): { min: number; max: number } | undefined {
  if (points.length === 0) return undefined
  return {
    min: points[0]!.time,
    max: points[points.length - 1]!.time,
  }
}

/**
 * Six signals as six panels, never one.
 *
 * A single chart with six series would need either six y-axes -- the dual-axis
 * fallacy, six times over -- or one axis shared between degrees Celsius,
 * millimetres per second, revolutions per minute, amperes, a dimensionless
 * load, and volts. The second is not a chart, it is a decoration: the relative
 * heights of the lines would encode the units they happen to be measured in
 * rather than anything about the machine.
 *
 * Every panel uses the same colour, because in small multiples each panel
 * holds one series and identity is carried by the panel's title. Six hues here
 * would be colour used to distinguish things that are already distinguished.
 */
export function buildTelemetryOption(series: TelemetrySeries): EChartsOption {
  const aggregated = !series.resolution.is_raw

  const titles = SENSOR_SIGNALS.map((signal, index) => ({
    text: `${signal} (${SIGNAL_UNITS[signal]})`,
    left: `${index % 2 === 0 ? 10 : 60}%`,
    top: `${4 + Math.floor(index / 2) * 32}%`,
    textStyle: { fontSize: 12, fontWeight: 'normal' as const, color: AXIS_LABEL_COLOUR },
  }))

  const grids = SENSOR_SIGNALS.map((_, index) => ({
    left: index % 2 === 0 ? '10%' : '60%',
    width: '34%',
    top: `${10 + Math.floor(index / 2) * 32}%`,
    height: '20%',
    containLabel: false,
  }))

  const axes = SENSOR_SIGNALS.map((signal, index) => {
    const points = seriesFor(series, signal)
    const bounds = extent(points)

    return {
      xAxis: {
        type: 'time' as const,
        gridIndex: index,
        min: bounds?.min,
        max: bounds?.max,
        axisLabel: { color: AXIS_LABEL_COLOUR, fontSize: 10, hideOverlap: true },
        axisLine: { lineStyle: { color: GRID_LINE_COLOUR } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value' as const,
        gridIndex: index,
        scale: true,
        axisLabel: {
          color: AXIS_LABEL_COLOUR,
          fontSize: 10,
          // An aggregated panel says so on its axis as well as in the badge.
          // A reader who screenshots one panel should not need the header to
          // know they are looking at averages.
          formatter: aggregated ? `{value}\nmean` : '{value}',
        },
        splitLine: { lineStyle: { color: GRID_LINE_COLOUR, type: 'solid' as const } },
      },
      series: [
        {
          type: 'line' as const,
          name: signal,
          xAxisIndex: index,
          yAxisIndex: index,
          // Markers only on an aggregated series, where they say "this point
          // is a bucket", and never on a raw one, where a thousand dots would
          // obscure the line they belong to.
          showSymbol: aggregated,
          symbolSize: 5,
          lineStyle: {
            width: LINE_WIDTH,
            // Dashing here means "these are aggregates", which is the one
            // place in this dashboard where a dashed line carries information
            // rather than decoration.
            type: aggregated ? ('dashed' as const) : ('solid' as const),
            color: SERIES_COLOUR,
          },
          itemStyle: { color: SERIES_COLOUR },
          // The third element is `sample_count`, which ECharts carries
          // alongside the point without plotting it. It reaches the tooltip,
          // which is where "how many readings is this mean resting on" can
          // actually be answered.
          data: points.map((point) => [point.time, point.value, point.sampleCount]),
        },
      ],
    }
  })

  return {
    title: titles,
    grid: grids,
    xAxis: axes.map((axis) => axis.xAxis),
    yAxis: axes.map((axis) => axis.yAxis),
    series: axes.flatMap((axis) => axis.series),
    // One crosshair across all six, so a hover reads the whole machine at one
    // instant rather than one signal at a time. This is the compensation for
    // splitting the signals up, and it is why small multiples are drawn as one
    // instance with six grids rather than six charts.
    axisPointer: { link: [{ xAxisIndex: 'all' }], label: { backgroundColor: '#16191f' } },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      confine: true,
      formatter: (params: unknown) => {
        const items = params as { seriesName: string; data: unknown }[]
        const lines = items.map((item) => {
          const [, value, sampleCount] = item.data as [number, number, number]
          // "mean of 12 readings" is the sentence that stops a bucketed point
          // being read as a measurement -- which is the whole reason the API
          // reports a resolution at all.
          const basis =
            sampleCount > 1 ? ` <span style="color:#5b6472">(mean of ${sampleCount})</span>` : ''
          return `${item.seriesName}: <b>${value.toFixed(2)}</b>${basis}`
        })
        const stamp = (items[0]?.data as [number, number, number] | undefined)?.[0]
        const when =
          stamp === undefined ? '' : new Date(stamp).toLocaleString(undefined, { timeZoneName: 'short' })
        return [`<div style="font-size:12px">${when}</div>`, ...lines].join('<br/>')
      },
    },
    animation: false,
  }
}

/**
 * Failure probability over time.
 *
 * The band each point belongs to is the server's, not this chart's: the
 * thresholds live in the API's configuration and are deliberately not
 * duplicated here. So the line is coloured per point from the `risk_level` the
 * API sent, rather than by shading regions this code would have to guess the
 * boundaries of.
 */
export function buildPredictionOption(predictions: readonly Prediction[]): EChartsOption {
  // Oldest first, so the line reads left to right in time.
  const ordered = [...predictions].sort(
    (left, right) => new Date(left.predicted_at).getTime() - new Date(right.predicted_at).getTime(),
  )

  // A plain array rather than a tuple: ECharts expects a mutable
  // `OptionDataValue[]`, and `as const` makes it readonly, which does not match.
  const points = ordered.map((prediction) => ({
    value: [new Date(prediction.predicted_at).getTime(), prediction.failure_probability],
    itemStyle: { color: riskColorVar(prediction.risk_level) },
  }))

  const times = points.map((point) => point.value[0])

  return {
    grid: { left: '8%', right: '4%', top: '8%', bottom: '16%', containLabel: true },
    xAxis: {
      type: 'time',
      min: times[0],
      max: times[times.length - 1],
      axisLabel: { color: AXIS_LABEL_COLOUR, hideOverlap: true },
      axisLine: { lineStyle: { color: GRID_LINE_COLOUR } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: {
        color: AXIS_LABEL_COLOUR,
        formatter: (value: number) => `${Math.round(value * 100)}%`,
      },
      splitLine: { lineStyle: { color: GRID_LINE_COLOUR } },
    },
    series: [
      {
        type: 'line',
        showSymbol: true,
        symbolSize: 6,
        lineStyle: { width: LINE_WIDTH, color: SERIES_COLOUR },
        data: points,
      },
    ],
    tooltip: {
      trigger: 'axis',
      confine: true,
      formatter: (params: unknown) => {
        const [first] = params as { data: { value: [number, number] } }[]
        if (first === undefined) return ''
        const [time, probability] = first.data.value
        return `${formatProbability(probability)} at ${new Date(time).toLocaleTimeString()}`
      },
    },
    animation: false,
  }
}

/**
 * The most severe band present, for a caption.
 *
 * Uses the API's own band names rather than inventing a wording, so the chart,
 * the tables and the incidents page all say the same word for the same thing.
 */
export function peakRisk(predictions: readonly Prediction[]): string | undefined {
  if (predictions.length === 0) return undefined
  const worst = [...predictions].sort(
    (left, right) => riskRank(right.risk_level) - riskRank(left.risk_level),
  )[0]
  return worst === undefined ? undefined : riskLabel(worst.risk_level)
}

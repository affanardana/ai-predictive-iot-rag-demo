/**
 * Telemetry over a chosen window.
 *
 * The window selector scopes all six panels at once, so the figure answers one
 * question -- "what has this machine been doing for the last five hours" --
 * rather than six.
 */

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { telemetryQuery } from '@/api/queries'
import { SENSOR_SIGNALS, SIGNAL_UNITS, type TelemetrySeries, type TimeWindow } from '@/api/types'
import { Card } from '@/components/Card'
import { ResolutionBadge } from '@/components/ResolutionBadge'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { buildTelemetryOption } from '@/components/charts/options'
import { useECharts } from '@/components/charts/useECharts'
import { formatInstant } from '@/lib/time'

/** The windows PRD section 8 requires, in the order it lists them. */
const WINDOWS: readonly TimeWindow[] = ['1h', '5h', '24h', '7d', '30d']

const WINDOW_LABELS: Record<TimeWindow, string> = {
  '1h': '1 hour',
  '5h': '5 hours',
  '24h': '24 hours',
  '7d': '7 days',
  '30d': '30 days',
}

export function TelemetryPanel({ machineId }: { machineId: string }) {
  const [window, setWindow] = useState<TimeWindow>('1h')
  const [showTable, setShowTable] = useState(false)
  const telemetry = useQuery(telemetryQuery(machineId, window))

  return (
    <Card
      title="Telemetry"
      subtitle="Six signals, each on its own scale."
      actions={
        <div className="flex items-center gap-2">
          {telemetry.data !== undefined && <ResolutionBadge resolution={telemetry.data.resolution} />}
          <div className="flex rounded border border-line">
            {WINDOWS.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setWindow(option)}
                className={`px-2.5 py-1 text-xs ${
                  option === window
                    ? 'bg-accent/10 font-medium text-accent'
                    : 'text-ink-muted hover:bg-canvas'
                }`}
              >
                {WINDOW_LABELS[option]}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setShowTable((previous) => !previous)}
            className="rounded border border-line px-2.5 py-1 text-xs text-ink-muted hover:bg-canvas"
          >
            {showTable ? 'Chart' : 'Table'}
          </button>
        </div>
      }
    >
      {telemetry.isPending && <LoadingState label="Loading telemetry…" />}
      {telemetry.isError && <ErrorState error={telemetry.error} />}
      {telemetry.data !== undefined &&
        (telemetry.data.points.length === 0 ? (
          <EmptyState>
            No readings in this window. The window ends at the machine's newest reading — where
            that is ahead of the wall clock, which happens while a simulation is being played back
            faster than real time, the chart follows the data rather than the clock.
          </EmptyState>
        ) : showTable ? (
          <TelemetryTable series={telemetry.data} />
        ) : (
          <TelemetryChart series={telemetry.data} />
        ))}
    </Card>
  )
}

function TelemetryChart({ series }: { series: TelemetrySeries }) {
  // Rebuilt only when the data changes. A fresh option object on every render
  // would have ECharts re-apply it continuously as frames arrive.
  const option = useMemo(() => buildTelemetryOption(series), [series])
  const container = useECharts(option)

  return (
    <div>
      {/* 520px holds three rows of two panels with their titles. */}
      <div ref={container} style={{ height: 520 }} />
      <p className="mt-2 text-xs text-ink-muted">
        Window ends {formatInstant(series.interval.end)} — the bounds actually queried, which may
        run ahead of the wall clock.
      </p>
    </div>
  )
}

function TelemetryTable({ series }: { series: TelemetrySeries }) {
  return (
    <div className="max-h-96 overflow-auto">
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-surface">
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-ink-muted">
            <th className="py-2 pr-4 font-medium">Time</th>
            {SENSOR_SIGNALS.map((signal) => (
              <th key={signal} className="py-2 pr-4 text-right font-medium">
                {signal} ({SIGNAL_UNITS[signal]})
              </th>
            ))}
            <th className="py-2 text-right font-medium">Samples</th>
          </tr>
        </thead>
        <tbody>
          {series.points.map((point) => (
            <tr key={point.timestamp} className="border-b border-line last:border-0">
              <td className="py-2 pr-4 whitespace-nowrap">{formatInstant(point.timestamp)}</td>
              {SENSOR_SIGNALS.map((signal) => (
                <td key={signal} className="py-2 pr-4 text-right">
                  {point.reading[signal].toFixed(2)}
                </td>
              ))}
              <td className="py-2 text-right text-ink-muted">{point.sample_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

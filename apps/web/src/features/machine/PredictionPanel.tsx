/**
 * How the model's estimate has moved, and the notice that it is an estimate.
 */

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'

import { predictionsQuery } from '@/api/queries'
import { Card } from '@/components/Card'
import { ModelOutputNotice } from '@/components/ModelOutputNotice'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { buildPredictionOption, peakRisk } from '@/components/charts/options'
import { useECharts } from '@/components/charts/useECharts'
import { formatProbability } from '@/lib/risk'
import { formatInstant } from '@/lib/time'
import type { Prediction } from '@/api/types'

export function PredictionPanel({ machineId }: { machineId: string }) {
  const predictions = useQuery(predictionsQuery(machineId))

  return (
    <Card
      title="Prediction history"
      subtitle="Failure probability over time, coloured by the risk band the API assigned."
    >
      {predictions.isPending && <LoadingState label="Loading predictions…" />}
      {predictions.isError && <ErrorState error={predictions.error} />}
      {predictions.data !== undefined &&
        (predictions.data.length === 0 ? (
          <EmptyState>
            No predictions yet. Scoring begins once 60 readings have been stored for this machine.
          </EmptyState>
        ) : (
          <PredictionHistory predictions={predictions.data} />
        ))}
    </Card>
  )
}

function PredictionHistory({ predictions }: { predictions: readonly Prediction[] }) {
  const option = useMemo(() => buildPredictionOption(predictions), [predictions])
  const container = useECharts(option)
  const peak = peakRisk(predictions)
  const latest = predictions[0]

  return (
    <div className="space-y-3">
      <div ref={container} style={{ height: 240 }} />

      {latest !== undefined && (
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-sm">
          <span>
            <span className="text-ink-muted">Latest </span>
            <span className="text-lg font-semibold">
              {formatProbability(latest.failure_probability)}
            </span>
            <span className="text-ink-muted"> at {formatInstant(latest.predicted_at)}</span>
          </span>
          {peak !== undefined && (
            <span className="text-ink-muted">
              Peak band seen: <span className="text-ink">{peak}</span>
            </span>
          )}
          <span className="text-ink-muted">
            Looking ahead {Math.round(latest.horizon_seconds / 60)} minutes
          </span>
          <span className="font-mono text-xs text-ink-muted">model {latest.model_version}</span>
        </div>
      )}

      {/* PRD section 9. Next to the number, not behind a tooltip. */}
      <ModelOutputNotice />
    </div>
  )
}

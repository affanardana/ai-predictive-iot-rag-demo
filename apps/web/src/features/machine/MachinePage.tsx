/**
 * Everything about one machine, on one screen.
 *
 * PRD section 20.3: "The user should not need to navigate through unrelated
 * screens to understand one machine." So current state, telemetry, trends,
 * prediction history and incidents are all here rather than behind tabs.
 */

import type { ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'

import { machineDetailQuery } from '@/api/queries'
import type { MachineSummary } from '@/api/types'
import { Card } from '@/components/Card'
import { ModelOutputNotice } from '@/components/ModelOutputNotice'
import { RiskBadge } from '@/components/RiskBadge'
import { StaleIndicator } from '@/components/StaleIndicator'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { IncidentTable } from '@/features/incidents/IncidentTable'
import { PredictionPanel } from '@/features/machine/PredictionPanel'
import { SimulationPanel } from '@/features/machine/SimulationPanel'
import { TelemetryPanel } from '@/features/machine/TelemetryPanel'
import { formatProbability } from '@/lib/risk'
import { formatInstant } from '@/lib/time'

export function MachinePage() {
  const { machineId } = useParams<{ machineId: string }>()

  if (machineId === undefined) {
    return <ErrorState error={new Error('No machine was named in the URL.')} />
  }

  return <MachineDetail machineId={machineId} />
}

function MachineDetail({ machineId }: { machineId: string }) {
  const detail = useQuery(machineDetailQuery(machineId))

  if (detail.isPending) return <LoadingState label="Loading machine…" />
  if (detail.isError) return <ErrorState error={detail.error} />

  const { summary, recent_incidents: incidents } = detail.data

  return (
    <div className="space-y-4">
      <div>
        <Link to="/machines" className="text-sm text-ink-muted hover:text-ink">
          ← Machines
        </Link>
        <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h1 className="text-2xl font-semibold">{summary.machine.machine_id}</h1>
          <span className="text-ink-muted">{summary.machine.name}</span>
          <RiskBadge level={summary.risk_level} />
        </div>
      </div>

      <CurrentState summary={summary} />

      {/* PRD §20.3 lists Simulation among the blocks this page must combine. */}
      <SimulationPanel machineId={machineId} />

      <TelemetryPanel machineId={machineId} />

      <PredictionPanel machineId={machineId} />

      <Card
        title="Incidents"
        subtitle="Raised when the model's band reaches High or Critical."
        actions={
          <Link to="/incidents" className="text-sm text-accent hover:underline">
            All incidents
          </Link>
        }
      >
        {incidents.length === 0 ? (
          <EmptyState>No incidents recorded for this machine.</EmptyState>
        ) : (
          <IncidentTable incidents={incidents} showMachine={false} />
        )}
      </Card>
    </div>
  )
}

function CurrentState({ summary }: { summary: MachineSummary }) {
  const prediction = summary.latest_prediction
  const telemetry = summary.latest_telemetry

  return (
    <Card title="Current state">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
        <Field label="Failure probability">
          {prediction === null ? (
            <span className="text-ink-muted">—</span>
          ) : (
            <span className="text-2xl font-semibold">
              {formatProbability(prediction.failure_probability)}
            </span>
          )}
        </Field>

        <Field label="Prediction horizon">
          {prediction === null ? (
            <span className="text-ink-muted">—</span>
          ) : (
            `${Math.round(prediction.horizon_seconds / 60)} min`
          )}
        </Field>

        <Field label="Open incidents">
          {summary.open_incident_count > 0 ? (
            <span className="font-medium text-risk-high">{summary.open_incident_count}</span>
          ) : (
            '0'
          )}
        </Field>

        <Field label="Reporting">
          <StaleIndicator
            recordedAt={telemetry?.recorded_at ?? null}
            isReporting={summary.is_reporting}
          />
        </Field>
      </dl>

      <p className="mt-4 text-xs text-ink-muted">
        Registered {formatInstant(summary.machine.registered_at)}
        {telemetry !== null && <> · latest reading {formatInstant(telemetry.recorded_at)}</>}
        {prediction !== null && <> · latest prediction {formatInstant(prediction.predicted_at)}</>}
      </p>

      {prediction === null && (
        <p className="mt-3 text-sm text-ink-muted">
          This machine has not been scored yet. The model needs 60 consecutive readings before it
          will produce a first prediction.
        </p>
      )}

      {prediction !== null && <ModelOutputNotice className="mt-3" />}
    </Card>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-ink-muted">{label}</dt>
      <dd className="mt-1">{children}</dd>
    </div>
  )
}

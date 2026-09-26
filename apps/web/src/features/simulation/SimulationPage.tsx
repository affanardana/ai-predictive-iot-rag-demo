/**
 * Starting and stopping simulated runs.
 *
 * Replaces the Phase 7 placeholder, whose docstring explained why it was not a
 * disabled form. What changed is that the backend now exists: `POST
 * /api/v1/simulations` reaches a simulator running in its own container, so
 * these controls do something.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { ApiError } from '@/api/errors'
import { fleetQuery, keys, resetSimulation, simulationsQuery, startSimulation, stopSimulation } from '@/api/queries'
import type { SimulationRun, SimulationScenario } from '@/api/types'
import { Card } from '@/components/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { RunList } from '@/features/simulation/RunList'
import { SimulationForm } from '@/features/simulation/SimulationForm'

export function SimulationPage() {
  const queryClient = useQueryClient()
  const fleet = useQuery(fleetQuery())
  const runs = useQuery(simulationsQuery())
  const [notice, setNotice] = useState<string | undefined>(undefined)

  /** Everything a run's lifecycle can change, refreshed together. */
  const invalidateRuns = () => {
    void queryClient.invalidateQueries({ queryKey: keys.simulations() })
    void queryClient.invalidateQueries({ queryKey: keys.all })
  }

  const start = useMutation({
    mutationFn: (request: {
      machine_id: string
      scenario: SimulationScenario
      seed: number | null
      duration_minutes: number
      demo: boolean
    }) => startSimulation(request),
    onSuccess: (run: SimulationRun) => {
      setNotice(`Started ${run.session_id} on ${run.machine_id}.`)
      invalidateRuns()
    },
    onError: (error) => setNotice(describe(error)),
  })

  const stop = useMutation({
    mutationFn: (sessionId: string) => stopSimulation(sessionId),
    onSuccess: invalidateRuns,
    onError: (error) => setNotice(describe(error)),
  })

  const reset = useMutation({
    mutationFn: (sessionId: string) => resetSimulation(sessionId),
    onSuccess: invalidateRuns,
    onError: (error) => setNotice(describe(error)),
  })

  const machines = fleet.data ?? []

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Simulation</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Runs execute in a container on the server. Nothing needs to be started on a laptop —
          which is what <span className="font-medium">AC-010</span> requires.
        </p>
      </div>

      {notice !== undefined && (
        <div className="rounded border border-line bg-surface px-4 py-2 text-sm">{notice}</div>
      )}

      {fleet.isPending && <LoadingState label="Loading machines…" />}
      {fleet.isError && <ErrorState error={fleet.error} />}

      {fleet.data !== undefined && (
        <Card title="Start a run" subtitle="A machine, a scenario, how much simulated time, and a seed.">
          {machines.length === 0 ? (
            <EmptyState>
              No machines are registered, so there is nothing to simulate. Register one first — the
              command is in <code className="font-mono text-xs">infra/compose/README.md</code>.
            </EmptyState>
          ) : (
            <SimulationForm
              machines={machines.map((summary) => summary.machine.machine_id)}
              busyRuns={runs.data ?? []}
              pending={start.isPending}
              onSubmit={(values) => {
                setNotice(undefined)
                start.mutate(values)
              }}
            />
          )}
        </Card>
      )}

      <Card
        title="Runs"
        subtitle="One run per machine, and the fleet may run several at once."
        actions={
          runs.data !== undefined && runs.data.length > 0 ? (
            <Link to="/machines" className="text-sm text-accent hover:underline">
              Watch a machine →
            </Link>
          ) : undefined
        }
      >
        {runs.isPending && <LoadingState label="Loading runs…" />}
        {runs.isError && <ErrorState error={runs.error} />}
        {runs.data !== undefined &&
          (runs.data.length === 0 ? (
            <EmptyState>
              No runs yet. A run takes a few minutes to watch and covers hours of simulated
              degradation; the first prediction arrives after 60 readings.
            </EmptyState>
          ) : (
            <RunList
              runs={runs.data}
              onStop={(sessionId) => {
                setNotice(undefined)
                stop.mutate(sessionId)
              }}
              onReset={(sessionId) => {
                setNotice(undefined)
                reset.mutate(sessionId)
              }}
              busy={stop.isPending || reset.isPending}
            />
          ))}
      </Card>
    </div>
  )
}

/** Turn a failure into something worth reading. */
function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return error instanceof Error ? error.message : 'That did not go through.'
}

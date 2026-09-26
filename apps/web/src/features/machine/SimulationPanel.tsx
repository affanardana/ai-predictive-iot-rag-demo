/**
 * A machine's simulation state, and a one-click way to start the demonstration.
 *
 * PRD §20.3 lists `Simulation` among the blocks a machine page must combine.
 * Split from the `/simulation` page deliberately: this is status plus the
 * single action a viewer most often wants, and the full form — scenario,
 * duration, seed — lives there. Putting the form here too would work against
 * the same section's instruction that a reader "should not need to navigate
 * through unrelated screens", by making this screen long enough to need
 * scrolling.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { ApiError } from '@/api/errors'
import { keys, machineSimulationsQuery, startSimulation, stopSimulation } from '@/api/queries'
import { Card } from '@/components/Card'
import { EmptyState, LoadingState } from '@/components/States'
import { SCENARIO_LABELS, describeDuration, describeRun, progressFraction } from '@/lib/simulation'

export function SimulationPanel({ machineId }: { machineId: string }) {
  const queryClient = useQueryClient()
  const runs = useQuery(machineSimulationsQuery(machineId))
  const [notice, setNotice] = useState<string | undefined>(undefined)

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: keys.simulations() })
    void queryClient.invalidateQueries({ queryKey: keys.machine(machineId) })
    void queryClient.invalidateQueries({ queryKey: keys.all })
  }

  const start = useMutation({
    mutationFn: () =>
      startSimulation({ machine_id: machineId, scenario: null, demo: true, seed: null, duration_minutes: 240 }),
    onSuccess: (run) => {
      setNotice(`Started ${run.session_id}.`)
      invalidate()
    },
    onError: (error) =>
      setNotice(error instanceof ApiError ? error.message : 'That did not go through.'),
  })

  const stop = useMutation({
    mutationFn: (sessionId: string) => stopSimulation(sessionId),
    onSuccess: invalidate,
    onError: (error) =>
      setNotice(error instanceof ApiError ? error.message : 'That did not go through.'),
  })

  const latest = runs.data?.[0]
  const active = runs.data?.find((run) => run.is_active)

  return (
    <Card
      title="Simulation"
      subtitle="Runs execute on the server, not on a laptop."
      actions={
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={active !== undefined || start.isPending}
            onClick={() => {
              setNotice(undefined)
              start.mutate()
            }}
            className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            {start.isPending ? 'Starting…' : 'Run the demonstration'}
          </button>
          {active !== undefined && (
            <button
              type="button"
              disabled={stop.isPending}
              onClick={() => {
                setNotice(undefined)
                stop.mutate(active.session_id)
              }}
              className="rounded border border-line px-3 py-1.5 text-sm hover:bg-canvas disabled:opacity-50"
            >
              Stop
            </button>
          )}
          <Link to="/simulation" className="text-sm text-accent hover:underline">
            Configure…
          </Link>
        </div>
      }
    >
      {runs.isPending && <LoadingState label="Loading runs…" />}

      {runs.data !== undefined &&
        (latest === undefined ? (
          <EmptyState>
            No runs recorded for this machine. The button above starts the canonical
            demonstration: bearing degradation, demo seed, 240 minutes of simulated time.
          </EmptyState>
        ) : (
          <div className="space-y-2 text-sm">
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
              <span className="font-mono text-xs text-ink-muted">{latest.session_id}</span>
              <span>{SCENARIO_LABELS[latest.scenario]}</span>
              <span className="text-ink-muted">{describeDuration(latest)} simulated</span>
              <span className="text-ink-muted">seed {latest.seed}</span>
            </div>

            <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
              <div
                className={`h-full ${latest.is_stale ? 'bg-risk-warning' : 'bg-accent'}`}
                style={{ width: `${Math.round(progressFraction(latest) * 100)}%` }}
              />
            </div>

            <p className="text-ink-muted">
              {describeRun(latest).label} — {latest.completed_ticks} of {latest.tick_count} readings.
            </p>

            {describeRun(latest).note !== undefined && (
              <p className="text-ink-muted">{describeRun(latest).note}</p>
            )}

            {/* The one hint that turns "the pipeline is broken somewhere between
                MQTT and the API" from a mystery into a message. No new field
                needed: it compares two things already on screen. */}
            {latest.is_active && latest.completed_ticks === 0 && (
              <p className="text-ink-muted">
                No readings have arrived yet. If this does not change, the pipeline between the
                broker and the API is the thing to check.
              </p>
            )}
          </div>
        ))}

      {notice !== undefined && <p className="mt-3 text-sm">{notice}</p>}
    </Card>
  )
}

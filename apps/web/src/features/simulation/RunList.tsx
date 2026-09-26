/**
 * Every run, with its progress and the two things an operator can do to it.
 *
 * Stop and Reset are separate buttons because they mean different things:
 * stopping ends a run and keeps the record of it, resetting discards the record
 * once the run has ended. Collapsing them would make "stop" quietly destroy the
 * history a demonstration exists to be reviewed from.
 */

import { Link } from 'react-router-dom'

import type { SimulationRun } from '@/api/types'
import { RUN_STATUS_LABELS, SCENARIO_LABELS, describeDuration, describeRun, progressFraction, simulatedInstantReached } from '@/lib/simulation'
import { formatInstant } from '@/lib/time'

export function RunList({
  runs,
  onStop,
  onReset,
  busy,
}: {
  runs: readonly SimulationRun[]
  onStop: (sessionId: string) => void
  onReset: (sessionId: string) => void
  busy: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[54rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-ink-muted">
            <th className="py-2 pr-4 font-medium">Machine</th>
            <th className="py-2 pr-4 font-medium">Scenario</th>
            <th className="py-2 pr-4 font-medium">Progress</th>
            <th className="py-2 pr-4 font-medium">Status</th>
            <th className="py-2 pr-4 font-medium">Seed</th>
            <th className="py-2 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const presentation = describeRun(run)
            return (
              <tr key={run.session_id} className="border-b border-line last:border-0 align-top">
                <td className="py-3 pr-4">
                  <Link to={`/machines/${run.machine_id}`} className="font-medium hover:text-accent">
                    {run.machine_id}
                  </Link>
                  <div className="font-mono text-xs text-ink-muted">{run.session_id}</div>
                </td>

                <td className="py-3 pr-4">
                  {SCENARIO_LABELS[run.scenario]}
                  <div className="text-xs text-ink-muted">
                    {describeDuration(run)} of simulated time
                  </div>
                </td>

                <td className="py-3 pr-4">
                  <Progress run={run} />
                </td>

                <td className="py-3 pr-4">
                  <StatusPill run={run} />
                  <div className="mt-0.5 text-xs text-ink-muted">{presentation.label}</div>
                  {presentation.note !== undefined && (
                    <div className="mt-1 max-w-xs text-xs text-ink-muted">{presentation.note}</div>
                  )}
                </td>

                <td className="py-3 pr-4 font-mono text-xs">{run.seed}</td>

                <td className="py-3">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      disabled={!run.is_active || busy}
                      onClick={() => onStop(run.session_id)}
                      className="rounded border border-line px-2 py-1 text-xs hover:bg-canvas disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Stop
                    </button>
                    <button
                      type="button"
                      disabled={run.is_active || busy}
                      onClick={() => onReset(run.session_id)}
                      title={
                        run.is_active
                          ? 'Stop the run before resetting it.'
                          : 'Discard this run’s record. Its telemetry is kept.'
                      }
                      className="rounded border border-line px-2 py-1 text-xs hover:bg-canvas disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Reset
                    </button>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Progress({ run }: { run: SimulationRun }) {
  const fraction = progressFraction(run)
  const percent = Math.round(fraction * 100)

  return (
    <div className="min-w-[10rem]">
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-line"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={`h-full ${run.is_stale ? 'bg-risk-warning' : 'bg-accent'}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="mt-1 text-xs text-ink-muted">
        {run.completed_ticks} of {run.tick_count} readings
      </div>
      {/* The simulated instant reached, which is what "how far into the
          scenario" actually means -- the readings cover hours of machine time
          regardless of how long the run has been on screen. */}
      <div className="text-xs text-ink-muted">reached {formatInstant(simulatedInstantReached(run).toISOString())}</div>
    </div>
  )
}

function StatusPill({ run }: { run: SimulationRun }) {
  const tone = run.is_stale
    ? 'bg-risk-warning/10 text-risk-warning'
    : run.status === 'RUNNING'
      ? 'bg-accent/10 text-accent'
      : run.status === 'COMPLETED'
        ? 'bg-risk-normal/10 text-risk-normal'
        : run.status === 'FAILED'
          ? 'bg-risk-critical/10 text-risk-critical'
          : 'bg-ink-muted/10 text-ink-muted'

  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${tone}`}>
      {RUN_STATUS_LABELS[run.status]}
      {run.is_stale && ' — not reporting'}
    </span>
  )
}

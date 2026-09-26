/**
 * Simulation runs, as the UI presents them.
 *
 * Pure functions, so the parts worth testing are testable without a DOM. The
 * label tables are `Record`s keyed on the generated unions: adding a scenario
 * or a status on the server fails `npm run typecheck` until it has a label
 * here, which is a compile-time coupling with no runtime array to drift.
 *
 * That is deliberately the opposite of `EVENT_KINDS` in `realtime/types.ts`,
 * which *must* be a hand-written array because a `text/event-stream` body
 * cannot be described in OpenAPI. Where the generator can see a union, it
 * should be the source; where it cannot, a test pins it instead.
 */

import type { RunStatus, SimulationRun, SimulationScenario } from '@/api/types'

export const SCENARIO_LABELS: Record<SimulationScenario, string> = {
  NORMAL: 'Normal operation',
  BEARING_DEGRADATION: 'Bearing degradation',
  OVERHEATING: 'Overheating',
  OVERLOAD: 'Overload',
}

export const RUN_STATUS_LABELS: Record<RunStatus, string> = {
  PENDING: 'Starting',
  RUNNING: 'Running',
  COMPLETED: 'Completed',
  STOPPED: 'Stopped',
  FAILED: 'Failed',
}

/** How far through a run is, as a fraction between 0 and 1. */
export function progressFraction(run: SimulationRun): number {
  if (run.tick_count <= 0) return 0
  return Math.min(1, run.completed_ticks / run.tick_count)
}

/**
 * The simulated instant a run has reached.
 *
 * What "how far into the scenario are we" actually means, and it is not the
 * same as progress: the readings cover simulated time, which advances by
 * `sample_interval_seconds` each tick and is unrelated to how long the run has
 * been on screen.
 */
export function simulatedInstantReached(run: SimulationRun): Date {
  const elapsedSeconds = run.completed_ticks * run.sample_interval_seconds
  return new Date(new Date(run.started_at).getTime() + elapsedSeconds * 1000)
}

/** How long the run covers, in words. */
export function describeDuration(run: SimulationRun): string {
  const minutes = Math.round(run.duration_seconds / 60)
  if (minutes < 60) return `${minutes} min`
  const hours = minutes / 60
  return Number.isInteger(hours) ? `${hours} h` : `${hours.toFixed(1)} h`
}

export interface RunPresentation {
  label: string
  /** Whether the run is producing readings right now. */
  live: boolean
  /** A sentence explaining anything unusual, or undefined. */
  note?: string
}

/**
 * A run's status in words, including the case the API cannot decide alone.
 *
 * `is_stale` is computed server-side because it needs a clock and a timeout.
 * What this adds is the sentence: "running but not reporting" is a different
 * situation from "running", and the difference is that the simulator service is
 * gone -- which a reader needs told rather than left to infer from a progress
 * bar that has stopped moving.
 */
export function describeRun(run: SimulationRun): RunPresentation {
  const label = RUN_STATUS_LABELS[run.status]

  if (run.is_stale) {
    return {
      label,
      live: false,
      note: 'The simulator has stopped reporting. It may have restarted or gone down.',
    }
  }

  if (run.status === 'FAILED' && run.detail !== null) {
    return { label, live: false, note: run.detail }
  }

  if (run.status === 'STOPPED' && run.detail !== null) {
    return { label, live: false, note: run.detail }
  }

  if (run.status === 'PENDING') {
    return { label, live: false, note: 'Waiting for the simulator to accept this run.' }
  }

  return { label, live: run.is_active }
}

/**
 * The shortest run the API accepts, in minutes.
 *
 * Mirrors `MINIMUM_DURATION_MINUTES` on the server, which is derived there from
 * the model's window rather than chosen. Duplicated as a constant so the form
 * can refuse the value before a round trip; the server refuses it too, and that
 * is the authority.
 */
export const MINIMUM_DURATION_MINUTES = 60

/** The durations the form offers, in simulated minutes. */
export const DURATION_PRESETS = [60, 120, 240, 480] as const

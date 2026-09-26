/**
 * The four fields PRD §11 names: machine, scenario, duration, seed/demo mode.
 *
 * Deliberately not a fifth field for playback speed. The spec does not list one,
 * and a caller setting a fast pace on a one-core box is the failure the
 * server's resource limits exist to bound — so the pace is a server setting and
 * the form does not offer it.
 */

import { useState, type ReactNode } from 'react'

import type { SimulationRun, SimulationScenario } from '@/api/types'
import { DURATION_PRESETS, MINIMUM_DURATION_MINUTES, SCENARIO_LABELS } from '@/lib/simulation'

const SCENARIOS = Object.keys(SCENARIO_LABELS) as SimulationScenario[]

/** The demonstration's machine, per PRD §12. */
const DEMO_MACHINE = 'M003'

export interface SimulationFormValues {
  machine_id: string
  scenario: SimulationScenario
  seed: number | null
  duration_minutes: number
  demo: boolean
}

export function SimulationForm({
  machines,
  busyRuns,
  pending,
  onSubmit,
}: {
  machines: readonly string[]
  busyRuns: readonly SimulationRun[]
  pending: boolean
  onSubmit: (values: SimulationFormValues) => void
}) {
  const [machineId, setMachineId] = useState(
    machines.includes(DEMO_MACHINE) ? DEMO_MACHINE : (machines[0] ?? ''),
  )
  const [scenario, setScenario] = useState<SimulationScenario>('BEARING_DEGRADATION')
  const [duration, setDuration] = useState<number>(240)
  const [demo, setDemo] = useState(true)
  const [seed, setSeed] = useState<string>('20260923')

  // Named rather than merely disabled, so a reader learns *why* the button is
  // unavailable instead of guessing at a broken form.
  const blocking = busyRuns.find((run) => run.machine_id === machineId && run.is_active)

  const parsedSeed = Number.parseInt(seed, 10)
  const seedValid = !Number.isNaN(parsedSeed)
  const durationValid = duration >= MINIMUM_DURATION_MINUTES
  const canSubmit = machineId !== '' && durationValid && (demo || seedValid) && !pending

  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault()
        if (!canSubmit) return
        onSubmit({
          machine_id: machineId,
          scenario,
          seed: demo ? null : parsedSeed,
          duration_minutes: duration,
          demo,
        })
      }}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Machine">
          <select
            value={machineId}
            onChange={(event) => setMachineId(event.target.value)}
            className={INPUT}
          >
            {machines.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Scenario">
          <select
            value={scenario}
            onChange={(event) => setScenario(event.target.value as SimulationScenario)}
            disabled={demo}
            className={INPUT}
          >
            {SCENARIOS.map((value) => (
              <option key={value} value={value}>
                {SCENARIO_LABELS[value]}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label="Duration"
          hint="Simulated time, not how long you watch. The model needs 60 readings before it predicts anything, so shorter runs cannot show a risk band at all."
        >
          <select
            value={duration}
            onChange={(event) => setDuration(Number(event.target.value))}
            disabled={demo}
            className={INPUT}
          >
            {DURATION_PRESETS.map((minutes) => (
              <option key={minutes} value={minutes}>
                {minutes} minutes
              </option>
            ))}
          </select>
        </Field>

        <Field label="Mode">
          <div className="flex items-center gap-4 pt-2">
            <label className="flex items-center gap-2 text-sm">
              <input type="radio" checked={demo} onChange={() => setDemo(true)} />
              Demo — a fixed seed, so the sequence is always the same
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="radio" checked={!demo} onChange={() => setDemo(false)} />
              Custom
            </label>
          </div>
        </Field>
      </div>

      {!demo && (
        <Field label="Seed" hint="The same seed reproduces the same telemetry, exactly.">
          <input
            type="number"
            value={seed}
            onChange={(event) => setSeed(event.target.value)}
            className={`${INPUT} max-w-xs`}
          />
        </Field>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {pending ? 'Starting…' : 'Start run'}
        </button>

        {blocking !== undefined && (
          <span className="text-sm text-ink-muted">
            {machineId} is already running{' '}
            <code className="font-mono text-xs">{blocking.session_id}</code>. Stop it first.
          </span>
        )}
      </div>
    </form>
  )
}

const INPUT =
  'w-full rounded border border-line bg-surface px-2 py-1.5 text-sm disabled:bg-canvas disabled:text-ink-muted'

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div>
      <label className="block text-sm font-medium">{label}</label>
      {hint !== undefined && <p className="mt-0.5 mb-1 text-xs text-ink-muted">{hint}</p>}
      <div className={hint === undefined ? 'mt-1' : ''}>{children}</div>
    </div>
  )
}

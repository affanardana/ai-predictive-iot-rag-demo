/**
 * What this is, for someone who arrived without being told.
 *
 * MASTERPLAN section 27 asks that a reviewer understand the project "without
 * requiring an explanation from the developer", and a portfolio piece is read
 * by people who did not watch it being built. Collapsed by default, because
 * the reader who already knows should not have to scroll past it every time --
 * but present on the landing page, which is where a stranger lands.
 */

import { useState } from 'react'

const PIPELINE = [
  ['Simulator', 'produces telemetry for a scenario'],
  ['MQTT', 'carries it to the orchestrator'],
  ['API', 'persists it, idempotently'],
  ['Model', 'scores the last 60 readings'],
  ['Risk band', 'derived from the probability'],
  ['Incident', 'raised when the band is High or Critical'],
] as const

export function AboutPanel() {
  const [open, setOpen] = useState(false)

  return (
    <section className="rounded-lg border border-line bg-surface">
      <button
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span className="text-sm font-medium">About this demonstration</span>
        <span aria-hidden="true" className="text-ink-muted">
          {open ? '−' : '+'}
        </span>
      </button>

      {open && (
        <div className="border-t border-line px-4 py-4 text-sm">
          <p className="text-ink-muted">
            A simulated fleet of industrial motors. Telemetry flows through a real pipeline and a
            trained model scores each machine continuously; nothing on this page is mock data.
          </p>

          <ol className="mt-4 space-y-1.5">
            {PIPELINE.map(([stage, description], index) => (
              <li key={stage} className="flex gap-2">
                <span className="w-4 shrink-0 text-right text-ink-muted">{index + 1}.</span>
                <span>
                  <span className="font-medium">{stage}</span>{' '}
                  <span className="text-ink-muted">{description}</span>
                </span>
              </li>
            ))}
          </ol>

          <p className="mt-4 text-ink-muted">
            The model is a binary classifier: it detects that failure risk is rising, and cannot
            say what is failing. That is why incidents are type{' '}
            <span className="font-medium text-ink">Unclassified</span>, and it is the gap the
            documentation retrieval work exists to close.
          </p>
        </div>
      )}
    </section>
  )
}

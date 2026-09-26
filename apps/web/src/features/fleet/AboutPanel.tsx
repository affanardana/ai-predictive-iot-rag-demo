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
  // Open on arrival, and collapsed only by choice.
  //
  // It used to start closed with a bare `+` on the right, which failed twice
  // over: a reader could not tell it was a control, and the one person it exists
  // for -- a reviewer arriving cold, who MASTERPLAN section 27 says must
  // understand the project without the developer explaining it -- was the one
  // who would never open it.
  const [open, setOpen] = useState(true)

  return (
    <section className="rounded-lg border border-line bg-surface">
      {/* The whole header is the click target -- a large one, which matters on
          a touch screen -- while the control on the right is what *looks*
          clickable. Styling only the text as a button would leave a small
          target; styling only the header as one, as it was, leaves something
          that reads as a heading because it is rendered exactly like every
          other panel title on the page.

          `cursor-pointer` is not decoration: Tailwind v4's preflight no longer
          sets it on buttons, so without this the pointer never changes and the
          control feels inert even to someone who has decided to click it. */}
      <button
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        aria-expanded={open}
        aria-controls="about-demonstration"
        className="flex w-full cursor-pointer items-center justify-between gap-3 rounded-lg px-4 py-3 text-left hover:bg-canvas"
      >
        <span className="flex items-center gap-2">
          <span
            aria-hidden="true"
            className={`text-xs text-accent transition-transform ${open ? 'rotate-90' : ''}`}
          >
            ▶
          </span>
          <span className="text-sm font-medium">About this demonstration</span>
        </span>
        <span className="rounded border border-line px-2 py-0.5 text-xs text-ink-muted">
          {open ? 'Hide' : 'Show'}
        </span>
      </button>

      {open && (
        <div id="about-demonstration" className="border-t border-line px-4 py-4 text-sm">
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

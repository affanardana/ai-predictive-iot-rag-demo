/**
 * What to do here, and what to watch while doing it.
 *
 * The companion to `AboutPanel`: that one says what the system is, this says
 * how to operate it and what it is meant to demonstrate. A reviewer arriving
 * cold needs both, and the second is the one they cannot infer — the pages are
 * ordinary dashboards until something is running.
 *
 * Written as a sequence of *actions and what each one shows*, because that is
 * the whole claim this project makes: not that a model can write a paragraph,
 * but that every sentence in it is traceable to something checkable.
 */

import { Link } from 'react-router-dom'

import { CollapsiblePanel } from '@/components/CollapsiblePanel'

const STEPS = [
  {
    title: 'Start a run',
    body: (
      <>
        <Link to="/machines/M003" className="text-accent underline underline-offset-2">
          M003
        </Link>{' '}
        &rarr; <em>Run the demonstration</em>, or the{' '}
        <Link to="/simulation" className="text-accent underline underline-offset-2">
          Simulation
        </Link>{' '}
        tab for the full form. It replays four hours of bearing degradation in about four
        minutes, one reading a second.
      </>
    ),
  },
  {
    title: 'Watch the risk climb',
    body: (
      <>
        The machine page shows the readings, the failure probability and the risk band, and
        raises an incident when the band crosses High. It moves without a refresh: the server
        pushes each change to the browser, so the charts follow the run as it happens.
      </>
    ),
  },
  {
    title: 'Ask the Copilot',
    body: (
      <>
        The{' '}
        <Link to="/copilot" className="text-accent underline underline-offset-2">
          Copilot
        </Link>{' '}
        tab has the demonstration question ready. An answer takes about twenty seconds, most of
        it spent watching the sources being read; what comes back separates what was{' '}
        <span className="font-medium text-ink">observed</span>, what was{' '}
        <span className="font-medium text-ink">predicted</span>, what the{' '}
        <span className="font-medium text-ink">documentation</span> says and what was{' '}
        <span className="font-medium text-ink">inferred</span> — with the evidence it was
        written from shown underneath it.
      </>
    ),
  },
  {
    title: 'Try to catch it out',
    body: (
      <>
        Ask for something the manuals do not cover — a torque specification, say. It refuses
        rather than inventing a procedure, and says which document came closest and how close
        that was.
      </>
    ),
  },
] as const

export function HowToPanel() {
  return (
    <CollapsiblePanel title="How to play" id="how-to-play">
      <p className="text-ink-muted">
        The fourth step is the one worth waiting for, and the last one is the point of the whole
        thing.
      </p>

      <ol className="mt-4 space-y-3">
        {STEPS.map((step, index) => (
          <li key={step.title} className="flex gap-3">
            <span className="w-4 shrink-0 text-right font-medium text-ink-muted">{index + 1}.</span>
            <span>
              <span className="font-medium">{step.title}</span>
              <span className="mt-0.5 block text-ink-muted">{step.body}</span>
            </span>
          </li>
        ))}
      </ol>

      <p className="mt-4 border-t border-line pt-4 text-ink-muted">
        The prose is written by a 1.5B model running on the same machine as everything else, and
        it reads like one — short sentences, and no flair. What is worth checking is the
        traceability: every number in an answer appeared in the evidence the model was given,
        every documented claim names a document, a version, a section and a page, and the
        activity beside it reports which sources were read before it started writing.
      </p>
    </CollapsiblePanel>
  )
}

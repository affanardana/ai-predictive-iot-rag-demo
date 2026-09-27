/**
 * The Copilot.
 *
 * A question goes in, and the page shows the answer being built: which sources
 * were consulted as they are read, then the prose as it is written, then the
 * evidence, citations and tool activity attached to it. On the demonstration
 * hardware an answer takes about twenty seconds, and streaming is what makes
 * that a demonstration rather than a page that looks broken.
 *
 * This page replaced `PhasePlaceholder`, which existed to say the route had no
 * backend yet. That statement stopped being true, so it went -- but the
 * vocabulary it carried did not: the four kinds of statement are still on the
 * page, now as the key to the evidence blocks above them rather than as a
 * promise about work to come.
 */

import { useEffect, useRef, useState } from 'react'

import { ApiError } from '@/api/errors'
import { askCopilot, type CopilotAnswer, type CopilotToolCall } from '@/api/stream'
import type { ConversationTurn } from '@/api/types'
import { Card } from '@/components/Card'
import { ErrorState } from '@/components/States'
import { AnswerBlock, ToolTrail } from '@/features/copilot/AnswerBlock'
import { EVIDENCE_VOCABULARY } from '@/features/copilot/vocabulary'

/** `PRD.md` AC-007's question, which the demonstration turns on. */
const EXAMPLE_QUESTION =
  'Why is M003 becoming risky and what should I inspect according to the SOP?'

/**
 * How many earlier turns are sent back with a question.
 *
 * Bounded because the prefill is half the wait on one core, and an unbounded
 * transcript would walk into the context window one follow-up at a time.
 */
const HISTORY_TURNS = 6

interface Pending {
  tools: CopilotToolCall[]
  text: string
}

export function CopilotPage() {
  const [question, setQuestion] = useState(EXAMPLE_QUESTION)
  const [answers, setAnswers] = useState<CopilotAnswer[]>([])
  const [pending, setPending] = useState<Pending | null>(null)
  const [error, setError] = useState<unknown>(null)
  const controller = useRef<AbortController | null>(null)

  // An answer costs twenty seconds of a single core, and the API cancels the
  // generation when the connection drops. Navigating away should therefore
  // abort rather than leave the box writing an answer nobody will read.
  useEffect(() => () => controller.current?.abort(), [])

  const busy = pending !== null
  const asked = question.trim() !== ''

  function ask() {
    const text = question.trim()
    if (text === '' || busy) return

    const request = new AbortController()
    controller.current = request
    setPending({ tools: [], text: '' })
    setError(null)

    askCopilot(
      { question: text, history: historyOf(answers) },
      (event) => {
        // `done` and `error` are the promise's business, not this callback's:
        // the answer is the resolved value, and a failure is thrown, so both
        // reach the same two places a caller already handles.
        setPending((previous) => {
          if (previous === null) return previous
          if (event.kind === 'tool') return { ...previous, tools: [...previous.tools, event.call] }
          if (event.kind === 'token') return { ...previous, text: previous.text + event.text }
          return previous
        })
      },
      request.signal,
    )
      .then((answer) => setAnswers((previous) => [answer, ...previous]))
      .catch((failure: unknown) => {
        // An abort is the reader leaving, not a failure to report.
        if (failure instanceof DOMException && failure.name === 'AbortError') return
        setError(failure)
      })
      .finally(() => {
        setPending(null)
        controller.current = null
      })
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Copilot</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Ask about a machine in plain language. The system decides which sources the question
          needs, reads them, and gives the evidence to a language model running on this machine —
          no hosted API. What comes back is the answer together with the evidence it was written
          from, so a claim can be checked against its source.
        </p>
      </div>

      <Card title="Ask a question">
        <form
          className="space-y-3"
          onSubmit={(event) => {
            event.preventDefault()
            ask()
          }}
        >
          <label className="block text-sm font-medium" htmlFor="copilot-question">
            Question
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              id="copilot-question"
              className="min-w-64 flex-1 rounded border border-line bg-canvas px-3 py-2 text-sm"
              value={question}
              maxLength={500}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Why is M003 becoming risky?"
            />
            <button
              type="submit"
              disabled={!asked || busy}
              className="rounded bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {busy ? 'Answering…' : 'Ask'}
            </button>
          </div>
        </form>

        <div className="mt-4">
          {pending !== null && <PendingAnswer pending={pending} />}
          {error !== null && <CopilotError error={error} />}
        </div>
      </Card>

      <Card
        title="Answers"
        subtitle="Newest first. Each answer carries the question it was asked, so the transcript reads in any order."
        actions={
          answers.length > 0 ? (
            <button
              type="button"
              className="text-sm text-accent underline underline-offset-2"
              onClick={() => setAnswers([])}
            >
              Clear
            </button>
          ) : undefined
        }
      >
        {answers.length === 0 ? (
          <p className="py-6 text-center text-sm text-ink-muted">
            No questions asked yet.
          </p>
        ) : (
          <div className="space-y-4">
            {answers.map((answer, index) => (
              <AnswerBlock key={`${answer.question}-${index}`} answer={answer} />
            ))}
          </div>
        )}
      </Card>

      <Card
        title="The four kinds of statement"
        subtitle="Every answer above separates its evidence this way."
      >
        <dl className="grid gap-3 sm:grid-cols-2">
          {Object.entries(EVIDENCE_VOCABULARY).map(([kind, entry]) => (
            <div key={kind} className="rounded border border-line px-3 py-2">
              <dt className="text-sm font-medium">{entry.name}</dt>
              <dd className="mt-0.5 text-sm text-ink-muted">{entry.detail}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-sm text-ink-muted">
          An assistant that cannot say which of these it is doing will present a guess about a
          bearing in the same voice it uses for a measured temperature. The kinds are assigned by
          the system as the evidence is gathered, not by the model as it writes.
        </p>
      </Card>
    </div>
  )
}

/**
 * The answer as it is being produced.
 *
 * The activity comes first because that is the order the server sends it, and
 * because it is what makes the wait legible: a reader watching the trend tool
 * run knows the system is working on their question rather than on nothing.
 */
function PendingAnswer({ pending }: { pending: Pending }) {
  return (
    <div className="rounded border border-line bg-canvas px-4 py-3">
      {pending.tools.length > 0 && (
        <div className="mb-3">
          <h3 className="text-xs font-medium tracking-wide text-ink-muted uppercase">Consulted</h3>
          <div className="mt-2">
            <ToolTrail calls={pending.tools} />
          </div>
        </div>
      )}
      {pending.text === '' ? (
        <p className="text-sm text-ink-muted">
          Reading the machine’s records… an answer takes about twenty seconds on this hardware.
        </p>
      ) : (
        <p className="text-sm leading-relaxed whitespace-pre-line">
          {pending.text}
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-ink-muted align-text-bottom"
          />
        </p>
      )}
    </div>
  )
}

/**
 * A failure that is about the deployment, not about the question.
 *
 * The same distinction `KnowledgePage` draws: 503 means the service behind the
 * route is not answering, 409 means another answer is already being written.
 * Neither is "the machine is fine" and neither should look like an error in
 * the question that was asked.
 */
function CopilotError({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.code === 'chat_unavailable') {
    return (
      <div className="rounded border border-risk-warning/30 bg-risk-warning/5 px-4 py-3">
        <p className="text-sm font-medium text-risk-warning">
          The language model is not answering.
        </p>
        <p className="mt-1 text-sm text-ink-muted">
          This is a deployment problem, not a missing answer. The model runs on this machine and
          loads lazily, so the first question after a restart can fail while it is still coming up.
        </p>
      </div>
    )
  }

  if (error instanceof ApiError && error.code === 'chat_busy') {
    return (
      <div className="rounded border border-line bg-canvas px-4 py-3">
        <p className="text-sm font-medium">One answer at a time.</p>
        <p className="mt-1 text-sm text-ink-muted">
          Another question is already being answered. This machine has one core, so a second answer
          would halve the speed of the first — ask again in about twenty seconds.
        </p>
      </div>
    )
  }

  return <ErrorState error={error} />
}

/**
 * The earlier turns, oldest first.
 *
 * Built from the answers this page is already holding rather than stored
 * anywhere: PRD section 20.6 asks for a conversation, and a transcript that
 * survives a reload would need a session and a retention rule that nothing
 * asks for. Phase 11 owns persisting it, alongside the authentication that
 * would give it an owner.
 */
function historyOf(answers: CopilotAnswer[]): ConversationTurn[] {
  return answers
    .slice(0, HISTORY_TURNS / 2)
    .flatMap((answer) => [
      { role: 'user', content: answer.question },
      { role: 'assistant', content: answer.answer },
    ])
    .reverse()
}

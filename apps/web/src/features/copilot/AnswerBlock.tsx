/**
 * One answer, and everything behind it.
 *
 * Rendered together deliberately. PRD section 18 requires the evidence and its
 * sources to be visible and section 16 requires the kinds to be distinguishable
 * -- so the prose is never shown on its own. A reader who wants to check a
 * claim has the evidence the model was given, and the tool activity that
 * produced it, directly beneath it.
 *
 * The three verdicts render differently because they are different things: an
 * answer, a refusal (the evidence does not support one), and a fallback (the
 * evidence was fine and the model's prose was not). Showing the last two as an
 * empty answer would collapse them into the same failure.
 */

import { Link } from 'react-router-dom'

import type { CopilotAnswer, CopilotEvidence, CopilotToolCall } from '@/api/stream'
import { EVIDENCE_KINDS } from '@/api/stream'
import { EVIDENCE_VOCABULARY, TOOL_NAMES } from '@/features/copilot/vocabulary'

/** The tools a question consulted, with what each found. */
export function ToolTrail({ calls }: { calls: CopilotToolCall[] }) {
  return (
    <ol className="space-y-1">
      {calls.map((call, index) => (
        <li key={`${call.tool}-${index}`} className="flex flex-wrap items-baseline gap-x-2 text-sm">
          {/* The tool name is a fallback, not a translation: a server that
              grows an eighth tool should print its own name rather than a
              blank space, until the page is redeployed with a label for it. */}
          <span className="font-medium">{TOOL_NAMES[call.tool] ?? call.tool}</span>
          <span className="text-ink-muted">{call.summary}</span>
        </li>
      ))}
    </ol>
  )
}

export function AnswerBlock({ answer }: { answer: CopilotAnswer }) {
  return (
    <article className="rounded-lg border border-line bg-surface px-4 py-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">{answer.question}</p>
        {answer.machine_id !== null && (
          <Link
            to={`/machines/${answer.machine_id}`}
            className="text-xs text-accent underline underline-offset-2"
          >
            {answer.machine_id}
          </Link>
        )}
      </div>

      <div className="mt-3">
        <VerdictBody answer={answer} />
      </div>

      {answer.evidence.length > 0 && (
        <section className="mt-4 border-t border-line pt-3">
          <h3 className="text-xs font-medium tracking-wide text-ink-muted uppercase">Evidence</h3>
          <div className="mt-2 space-y-3">
            <EvidenceGroups evidence={answer.evidence} />
          </div>
        </section>
      )}

      {answer.tool_calls.length > 0 && (
        <section className="mt-4 border-t border-line pt-3">
          <h3 className="text-xs font-medium tracking-wide text-ink-muted uppercase">Consulted</h3>
          <div className="mt-2">
            <ToolTrail calls={answer.tool_calls} />
          </div>
        </section>
      )}

      {answer.citations.length > 0 && (
        <section className="mt-4 border-t border-line pt-3">
          <h3 className="text-xs font-medium tracking-wide text-ink-muted uppercase">Sources</h3>
          <ul className="mt-2 space-y-1">
            {answer.citations.map((citation) => (
              <li key={citation.label} className="text-sm">
                {citation.label}
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className="mt-4 border-t border-line pt-3 text-xs text-ink-muted">
        {answer.model_id === null
          ? 'No language model was called: this answer is the system’s own.'
          : `Evidence gathered and attached by the system. The prose was written by ${answer.model_id} from the evidence above, and nothing else.`}
      </p>
    </article>
  )
}

function VerdictBody({ answer }: { answer: CopilotAnswer }) {
  if (answer.verdict === 'REFUSED') {
    return (
      <div className="rounded border border-risk-warning/30 bg-risk-warning/5 px-4 py-3">
        <p className="text-sm font-medium text-risk-warning">
          The evidence does not answer this question.
        </p>
        <p className="mt-1 text-sm text-ink-muted">{answer.reason}</p>
        {answer.evidence.length > 0 && (
          <p className="mt-2 text-xs text-ink-muted">
            What the system did find is shown below. It is not an answer to the question asked.
          </p>
        )}
      </div>
    )
  }

  if (answer.verdict === 'FALLBACK') {
    return (
      <div>
        <div className="rounded border border-risk-warning/30 bg-risk-warning/5 px-4 py-3">
          <p className="text-sm font-medium text-risk-warning">
            The model’s answer failed its grounding check.
          </p>
          <p className="mt-1 text-sm text-ink-muted">
            It stated {sentenceList(answer.ungrounded)}, which appear nowhere in the evidence it was
            given. The evidence is rendered below instead.
          </p>
        </div>
        <p className="mt-3 text-sm leading-relaxed whitespace-pre-line">{answer.answer}</p>
      </div>
    )
  }

  return <p className="text-sm leading-relaxed whitespace-pre-line">{answer.answer}</p>
}

/**
 * The evidence, grouped by kind.
 *
 * Iterated over the vocabulary rather than over the evidence, so the groups
 * always appear in the same order and a reader learns where to look. Grouping
 * is the rendering of PRD section 16's rule: a predicted value must not be
 * read as a measured one, and a list sorted by retrieval order would put them
 * side by side.
 */
function EvidenceGroups({ evidence }: { evidence: CopilotEvidence[] }) {
  return (
    <>
      {EVIDENCE_KINDS.map((kind) => {
        const items = evidence.filter((item) => item.kind === kind)
        if (items.length === 0) return null
        return (
          <div key={kind}>
            <h4 className="text-xs font-medium text-ink-muted" title={EVIDENCE_VOCABULARY[kind].detail}>
              {EVIDENCE_VOCABULARY[kind].name}
            </h4>
            <ul className="mt-1 space-y-1">
              {items.map((item, index) => (
                <li key={`${kind}-${index}`} className="text-sm">
                  {item.text}
                  {item.source !== null && (
                    <span className="block text-xs text-ink-muted">{item.source}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )
      })}
    </>
  )
}

function sentenceList(values: string[]): string {
  if (values.length === 1) return values[0]
  return `${values.slice(0, -1).join(', ')} and ${values[values.length - 1]}`
}

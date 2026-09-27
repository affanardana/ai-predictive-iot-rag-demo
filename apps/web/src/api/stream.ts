/**
 * Asking the Copilot, and reading the answer as it arrives.
 *
 * **The second place the "never hand-write the client" rule is broken**, and
 * for the reason `src/realtime/types.ts` gives: the answer travels inside a
 * `text/event-stream` body, which OpenAPI cannot describe, so it is not in
 * `schema.d.ts` and cannot be generated. The request *is* generated -- the
 * question and the history come from the document -- and that is exactly where
 * the line falls. What a client sends is describable; what a stream sends back
 * is not.
 *
 * Two tests make that safe rather than merely convenient:
 * `apps/api/tests/integration/test_copilot_api.py` pins the frame names and
 * every enum value on the server, and `stream.test.ts` pins the same ones here.
 * A rename on either side fails a test rather than freezing a page.
 *
 * A `POST`, so `EventSource` cannot be used: it only issues `GET` and cannot
 * carry a body. Frames are read from the response body by hand.
 *
 * It does not go through `api/client.ts` either, and for the same kind of
 * reason: that module awaits a JSON body, and the whole point here is to read
 * the body while it is still arriving. The path is therefore not in `ApiPath`,
 * which exists to make the non-streaming calls typo-proof.
 */

import { API_BASE_URL } from '@/config'
import { ApiError, toApiError } from '@/api/errors'
import type { AskRequest, KnowledgeCitation } from '@/api/types'

/**
 * The four kinds of statement an answer separates.
 *
 * `PRD.md` section 16's list, and the reason the evidence blocks are rendered
 * at all. A `const` tuple with the union derived from it, so the page can
 * iterate the kinds to group evidence -- and so a fifth kind added on the
 * server, once mirrored here, is a compile error everywhere it is switched on
 * rather than an evidence block that silently stops appearing.
 */
export const EVIDENCE_KINDS = ['OBSERVED', 'PREDICTED', 'DOCUMENTED', 'INFERRED'] as const

export type EvidenceKind = (typeof EVIDENCE_KINDS)[number]

/** What the Copilot did with the question. */
export const ANSWER_VERDICTS = ['ANSWERED', 'REFUSED', 'FALLBACK'] as const

export type AnswerVerdict = (typeof ANSWER_VERDICTS)[number]

/** The names `MASTERPLAN.md` section 6 gives the tools, as the frames carry them. */
export const COPILOT_TOOLS = [
  'get_machine_current_state',
  'get_machine_telemetry',
  'get_machine_trend',
  'get_machine_prediction',
  'get_machine_incidents',
  'get_machine_history',
  'search_maintenance_knowledge',
] as const

export type CopilotTool = (typeof COPILOT_TOOLS)[number]

/** One finding, with the provenance that justifies it. */
export interface CopilotEvidence {
  kind: EvidenceKind
  text: string
  /** Set exactly when the kind is DOCUMENTED, which the domain enforces. */
  source: string | null
}

/** One tool's execution, for the activity trail. */
export interface CopilotToolCall {
  tool: CopilotTool
  summary: string
  evidence_count: number
}

/** A finished answer and everything behind it. */
export interface CopilotAnswer {
  question: string
  verdict: AnswerVerdict
  answer: string
  machine_id: string | null
  evidence: CopilotEvidence[]
  citations: KnowledgeCitation[]
  tool_calls: CopilotToolCall[]
  /** Why the answer is a refusal, when it is one. */
  reason: string
  /**
   * Numbers the model stated that appear in none of the evidence it was given.
   *
   * Non-empty is what turns an answer into a fallback. Surfaced rather than
   * hidden: it is the one signal that says the prose is not to be trusted even
   * though the evidence is.
   */
  ungrounded: string[]
  model_id: string | null
}

/** A frame as it arrives, in order. */
export type CopilotEvent =
  | { kind: 'tool'; call: CopilotToolCall }
  | { kind: 'token'; text: string }
  | { kind: 'done'; answer: CopilotAnswer }
  | { kind: 'error'; error: ApiError }

/** One server-sent event, before it is interpreted. */
export interface Frame {
  event: string
  data: string
}

/**
 * A blank line ends a frame, either line ending permitted.
 *
 * A regex rather than `indexOf('\n\n')` so a server using CRLF is handled
 * without normalising the buffer first -- a normalisation that would itself
 * break if a chunk boundary landed between the `\r` and the `\n`.
 */
const SEPARATOR = /\r?\n\r?\n/

/**
 * A reader over the response body's chunks.
 *
 * Stateful, because the bytes do not respect frame boundaries: a chunk can end
 * mid-frame and the remainder has to be held until the next chunk completes it.
 * That is the classic server-sent-events bug, and it is why this is a separate
 * function over strings -- the split points can be chosen in a test rather than
 * waited for.
 */
export function createFrameReader(): (chunk: string) => Frame[] {
  let buffer = ''
  return (chunk) => {
    buffer += chunk
    const frames: Frame[] = []
    for (;;) {
      const match = SEPARATOR.exec(buffer)
      if (match === null) break
      const block = buffer.slice(0, match.index)
      buffer = buffer.slice(match.index + match[0].length)
      const frame = parseFrame(block)
      if (frame !== null) frames.push(frame)
    }
    return frames
  }
}

function parseFrame(block: string): Frame | null {
  let event = ''
  const data: string[] = []
  for (const line of block.split('\n')) {
    // A comment. The API sends them as keepalives while a tool is running, and
    // they carry nothing.
    if (line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim()
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).trimStart())
  }
  if (event === '' || data.length === 0) return null
  return { event, data: data.join('\n') }
}

/** Turn a raw frame into an event, or `null` if it is not one this client knows. */
function toEvent(frame: Frame): CopilotEvent | null {
  const body = JSON.parse(frame.data) as Record<string, unknown>
  switch (frame.event) {
    case 'tool':
      return { kind: 'tool', call: body as unknown as CopilotToolCall }
    case 'token':
      return { kind: 'token', text: String(body.text ?? '') }
    case 'done':
      return { kind: 'done', answer: body as unknown as CopilotAnswer }
    case 'error':
      return {
        kind: 'error',
        error: new ApiError(
          Number(body.status ?? 500),
          String(body.code ?? 'stream_error'),
          String(body.message ?? 'The answer failed.'),
        ),
      }
    default:
      return null
  }
}

/**
 * Ask a question and report the answer as it is produced.
 *
 * `onEvent` is called for every frame in the order the server sent it: the tool
 * activity first, then the answer's tokens, then the finished answer. The
 * return value is that same finished answer, so a caller that only wants the
 * result can ignore the callback and await this.
 *
 * Failures arrive on two channels, and deliberately. A busy Copilot or an
 * over-long question is refused before the stream starts, so it is a status
 * code and `toApiError` turns it into the same `ApiError` every other call in
 * the app throws. Everything after that point is a `200` with the status
 * already committed, so a failure has to be a frame -- and it is turned back
 * into an `ApiError` here, because where it failed is the server's business and
 * not the caller's.
 */
export async function askCopilot(
  body: AskRequest,
  onEvent: (event: CopilotEvent) => void,
  signal?: AbortSignal,
): Promise<CopilotAnswer> {
  const response = await fetch(`${API_BASE_URL}/api/v1/copilot/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) throw await toApiError(response)
  if (response.body === null) {
    throw new ApiError(response.status, 'unreadable_stream', 'The answer stream had no body.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const readFrames = createFrameReader()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      // `stream: true` because a chunk boundary can fall inside a multi-byte
      // character; decoding each chunk on its own would corrupt the answer
      // rather than merely delay it.
      for (const frame of readFrames(decoder.decode(value, { stream: true }))) {
        const event = toEvent(frame)
        // An unrecognised frame is dropped rather than thrown: a newer server
        // sending something this build does not know must not cost a reader the
        // twenty seconds already spent on their answer. A `done` frame that
        // never arrives is still reported, below.
        if (event === null) continue
        onEvent(event)
        if (event.kind === 'done') return event.answer
        if (event.kind === 'error') throw event.error
      }
    }
  } finally {
    // Returning early from inside the loop leaves the body unread, and an
    // unread body holds the connection open. Cancelling is a no-op once the
    // stream has ended, and an error here is not worth masking the real one.
    await reader.cancel().catch(() => undefined)
  }

  throw new ApiError(response.status, 'stream_incomplete', 'The stream ended before an answer did.')
}

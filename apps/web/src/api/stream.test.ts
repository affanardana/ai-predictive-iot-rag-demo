import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '@/api/errors'
import {
  ANSWER_VERDICTS,
  COPILOT_TOOLS,
  EVIDENCE_KINDS,
  askCopilot,
  createFrameReader,
  type CopilotEvent,
} from '@/api/stream'

/** One frame, in the exact form `apps/api/.../routers/copilot.py` writes it. */
function frame(event: string, body: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(body)}\n\n`
}

/**
 * A recorded answer, field for field.
 *
 * The same payload `test_the_final_frame_carries_the_evidence_and_its_sources`
 * asserts on the server. Renaming a field there fails a test here rather than
 * leaving a page that renders nothing -- which is the whole reason these types
 * are hand-written.
 */
const ANSWER = {
  question: 'Why is M003 becoming risky?',
  verdict: 'ANSWERED',
  answer: 'The failure probability is 0.81 [1].',
  machine_id: 'M003',
  evidence: [
    { kind: 'OBSERVED', text: 'vibration 2.31 mm/s', source: null },
    {
      kind: 'DOCUMENTED',
      text: 'Attach an accelerometer to the bearing housing.',
      source: 'Bearing Inspection SOP v1.4, section 3, page 1',
    },
  ],
  citations: [
    {
      document_key: 'bearing-inspection-sop',
      title: 'Bearing Inspection SOP',
      version: '1.4',
      section: '3',
      page: 1,
      label: 'Bearing Inspection SOP v1.4, section 3, page 1',
    },
  ],
  tool_calls: [
    { tool: 'get_machine_trend', summary: 'vibration rising 1.4 → 2.3 mm/s', evidence_count: 2 },
  ],
  reason: '',
  ungrounded: [],
  model_id: 'qwen2.5-1.5b-instruct-q4_k_m',
}

function streamOf(...chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(typeof chunk === 'string' ? encoder.encode(chunk) : chunk)
      }
      controller.close()
    },
  })
}

function respond(...chunks: (string | Uint8Array)[]): Response {
  return new Response(streamOf(...chunks), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function stubFetch(response: Response) {
  const fetchMock = vi.fn(async () => response)
  // The cast is the test's: what matters is that `askCopilot` reads `ok`,
  // `status`, `body` and, for a failure, `json` -- all of which a real
  // `Response` provides and this one does too.
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('createFrameReader', () => {
  it('reads a frame delivered whole', () => {
    const read = createFrameReader()

    expect(read(frame('token', { text: 'Vibration' }))).toEqual([
      { event: 'token', data: '{"text":"Vibration"}' },
    ])
  })

  it('holds a frame whose chunk boundary lands inside it', () => {
    // The classic bug, and the reason this is a function of its own: a chunk
    // boundary falls wherever the network puts it, so half a frame arrives,
    // and a parser that treats each chunk as a frame loses the answer.
    const read = createFrameReader()

    expect(read('event: token\nda')).toEqual([])
    expect(read('ta: {"text":"Vibration"}\n')).toEqual([])
    expect(read('\n')).toEqual([{ event: 'token', data: '{"text":"Vibration"}' }])
  })

  it('reads several frames from one chunk', () => {
    const read = createFrameReader()

    const frames = read(
      frame('tool', { tool: 'get_machine_trend' }) + frame('token', { text: 'Vibration' }),
    )

    expect(frames.map((item) => item.event)).toEqual(['tool', 'token'])
  })

  it('splits on a blank line written with CRLF', () => {
    // A proxy is free to rewrite line endings, and the failure is silent: the
    // frames never split, so nothing arrives at all.
    const read = createFrameReader()

    expect(read('event: token\r\ndata: {"text":"a"}\r\n\r\n')).toEqual([
      { event: 'token', data: '{"text":"a"}' },
    ])
  })

  it('discards a keepalive comment', () => {
    // The API sends these while a tool is running, to hold the connection
    // open. They carry nothing and must not become an event.
    const read = createFrameReader()

    expect(read(': keepalive\n\n')).toEqual([])
    expect(read(frame('token', { text: 'a' }))).toEqual([{ event: 'token', data: '{"text":"a"}' }])
  })

  it('discards a block with no data line', () => {
    const read = createFrameReader()

    expect(read('event: token\n\n')).toEqual([])
  })
})

describe('askCopilot', () => {
  it('reports the tools, then the tokens, then the answer', async () => {
    // The frame order is the feature: twenty seconds of silence reads as a
    // broken page, and the activity is what makes the wait legible.
    stubFetch(
      respond(
        frame('tool', { tool: 'get_machine_trend', summary: 'rising', evidence_count: 2 }),
        frame('token', { text: 'The ' }),
        frame('token', { text: 'failure ' }),
        frame('done', ANSWER),
      ),
    )
    const events: CopilotEvent[] = []

    const answer = await askCopilot({ question: 'Why is M003 risky?' }, (event) =>
      events.push(event),
    )

    expect(events.map((event) => event.kind)).toEqual(['tool', 'token', 'token', 'done'])
    expect(answer.answer).toBe('The failure probability is 0.81 [1].')
  })

  it('returns everything the server attached to the answer', async () => {
    stubFetch(respond(frame('done', ANSWER)))

    const answer = await askCopilot({ question: 'Why is M003 risky?' }, () => undefined)

    expect(answer.citations[0].label).toBe('Bearing Inspection SOP v1.4, section 3, page 1')
    expect(answer.evidence[1].source).toBe('Bearing Inspection SOP v1.4, section 3, page 1')
    // Every field the server sends, unrenamed. An exact key set rather than a
    // spot check: a field added to the server and not mirrored here is a field
    // the page will silently never show.
    expect(Object.keys(answer).sort()).toEqual([
      'answer',
      'citations',
      'evidence',
      'machine_id',
      'model_id',
      'question',
      'reason',
      'tool_calls',
      'ungrounded',
      'verdict',
    ])
  })

  it('reads a frame split across two chunks of the response body', async () => {
    const whole = frame('done', ANSWER)
    const cut = Math.floor(whole.length / 2)
    stubFetch(respond(whole.slice(0, cut), whole.slice(cut)))

    const answer = await askCopilot({ question: 'Why is M003 risky?' }, () => undefined)

    expect(answer.verdict).toBe('ANSWERED')
  })

  it('reassembles a character split across two chunks', async () => {
    // `TextDecoder.decode` without `stream: true` would corrupt this: the
    // answer would arrive with a replacement character where the degree sign
    // was, and only for answers unlucky enough to straddle a buffer.
    const encoder = new TextEncoder()
    const bytes = encoder.encode(frame('token', { text: '48.2 °C' }))
    const cut = bytes.indexOf(0xc2) + 1
    stubFetch(respond(bytes.slice(0, cut), bytes.slice(cut)))
    const tokens: string[] = []

    await askCopilot({ question: 'Temperature?' }, (event) => {
      if (event.kind === 'token') tokens.push(event.text)
    }).catch(() => undefined)

    expect(tokens.join('')).toBe('48.2 °C')
  })

  it('drops a frame it does not recognise', async () => {
    // A newer server sending something this build does not know must not cost
    // the reader the twenty seconds already spent. A `done` that never arrives
    // is still reported, so this cannot hide a renamed frame for long.
    stubFetch(
      respond(frame('thinking', { note: 'new in a later release' }), frame('done', ANSWER)),
    )

    const answer = await askCopilot({ question: 'Why is M003 risky?' }, () => undefined)

    expect(answer.verdict).toBe('ANSWERED')
  })

  it('turns a failure frame into the error the envelope would have carried', async () => {
    // The status was committed when the response began, so a failure after
    // that point arrives in band -- with the same code and status, so the page
    // handles it exactly as it handles a 503 from any other route.
    stubFetch(
      respond(
        frame('tool', { tool: 'get_machine_trend', summary: 'rising', evidence_count: 2 }),
        frame('error', { code: 'chat_unavailable', status: 503, message: 'The model is loading.' }),
      ),
    )
    const events: CopilotEvent[] = []

    const failure = await askCopilot({ question: 'Why is M003 risky?' }, (event) =>
      events.push(event),
    ).catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(ApiError)
    expect(failure as ApiError).toMatchObject({
      code: 'chat_unavailable',
      status: 503,
      message: 'The model is loading.',
    })
    // The activity that did happen is still reported: a reader who watched two
    // tools run should not have the page forget them when the third failed.
    expect(events.map((event) => event.kind)).toEqual(['tool', 'error'])
  })

  it('throws the envelope when the refusal precedes the stream', async () => {
    // A busy Copilot is a real status code: nothing has been sent yet, so it
    // does not need to be a frame.
    stubFetch(
      new Response(
        JSON.stringify({
          error: { code: 'chat_busy', message: 'Another answer is being written.', details: null },
        }),
        { status: 409, headers: { 'Content-Type': 'application/json' } },
      ),
    )

    const failure = await askCopilot({ question: 'Why is M003 risky?' }, () => undefined).catch(
      (error: unknown) => error,
    )

    expect(failure).toBeInstanceOf(ApiError)
    expect((failure as ApiError).code).toBe('chat_busy')
    expect((failure as ApiError).isConflict).toBe(true)
  })

  it('throws when the stream ends without an answer', async () => {
    stubFetch(respond(frame('token', { text: 'The ' })))

    const failure = await askCopilot({ question: 'Why is M003 risky?' }, () => undefined).catch(
      (error: unknown) => error,
    )

    expect(failure).toBeInstanceOf(ApiError)
    expect((failure as ApiError).code).toBe('stream_incomplete')
  })
})

describe('the mirrored vocabularies', () => {
  it('names every kind, verdict and tool the server can send', () => {
    // Asserted literally because these cannot be generated. The server-side
    // twin is `test_the_copilot_contract_is_stable` in
    // `apps/api/tests/integration/test_copilot_api.py`, which asserts the same
    // lists from the enums -- so a value renamed on the server fails there, and
    // one renamed here fails this file. The pair is what replaces generation.
    expect(EVIDENCE_KINDS).toEqual(['OBSERVED', 'PREDICTED', 'DOCUMENTED', 'INFERRED'])
    expect(ANSWER_VERDICTS).toEqual(['ANSWERED', 'REFUSED', 'FALLBACK'])
    expect(COPILOT_TOOLS).toEqual([
      'get_machine_current_state',
      'get_machine_telemetry',
      'get_machine_trend',
      'get_machine_prediction',
      'get_machine_incidents',
      'get_machine_history',
      'search_maintenance_knowledge',
    ])
  })
})

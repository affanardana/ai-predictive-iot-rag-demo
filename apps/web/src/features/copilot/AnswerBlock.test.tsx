import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import type { CopilotAnswer } from '@/api/stream'
import { AnswerBlock } from '@/features/copilot/AnswerBlock'

/**
 * What a reader actually sees for each verdict.
 *
 * **The first tests in this project that render a component.** `vite.config.ts`
 * says rendering is not tested here, and that was true while Playwright was
 * planned for Phase 11; the owner declined it, so this tier is what covers the
 * page instead. It is not a substitute for a browser -- nothing here exercises
 * routing, styling or ECharts -- but every defect Phase 10 found on the
 * deployed stack was in *what the page said*, and that is exactly what this
 * checks.
 */

function anAnswer(overrides: Partial<CopilotAnswer> = {}): CopilotAnswer {
  return {
    question: 'Why is M003 becoming risky?',
    verdict: 'ANSWERED',
    answer: 'The failure probability is 0.81 [1].',
    machine_id: 'M003',
    evidence: [],
    citations: [],
    tool_calls: [],
    reason: '',
    ungrounded: [],
    model_id: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    ...overrides,
  }
}

function renderAnswer(answer: CopilotAnswer) {
  return render(
    <MemoryRouter>
      <AnswerBlock answer={answer} />
    </MemoryRouter>,
  )
}

describe('AnswerBlock', () => {
  it('shows the prose and attributes it to the model', () => {
    renderAnswer(anAnswer())

    expect(screen.getByText(/failure probability is 0.81/)).toBeTruthy()
    expect(screen.getByText(/qwen2\.5-1\.5b-instruct-q4_k_m\.gguf/)).toBeTruthy()
  })

  it('renders a refusal as a refusal, with the reason and no prose', () => {
    // The distinction the whole evidence model exists for: a refusal is not an
    // empty answer, and rendering it as one would collapse "the documentation
    // does not cover this" into "something went wrong".
    renderAnswer(
      anAnswer({
        verdict: 'REFUSED',
        answer: 'The available documentation is insufficient to support that recommendation.',
        reason: 'The closest maintenance document scored -6.71, below the -3.30 needed.',
      }),
    )

    expect(screen.getByText(/does not answer this question/i)).toBeTruthy()
    expect(screen.getByText(/scored -6.71/)).toBeTruthy()
    expect(screen.queryByText(/insufficient to support that recommendation/)).toBeNull()
  })

  it('says a refusal came from the system rather than the model', () => {
    // "No language model was called" is a claim about how the answer was
    // produced, and the footer is the only place it can be made.
    renderAnswer(anAnswer({ verdict: 'REFUSED', answer: 'x', model_id: null }))

    expect(screen.getByText(/No language model was called/)).toBeTruthy()
  })

  it('names the numbers that failed the grounding check', () => {
    // A fallback that did not say which number was invented would be a
    // fallback a reader cannot check.
    renderAnswer(
      anAnswer({
        verdict: 'FALLBACK',
        answer: 'The findings are listed below.',
        ungrounded: ['0.81', '4.2'],
      }),
    )

    expect(screen.getByText(/failed its grounding check/i)).toBeTruthy()
    expect(screen.getByText(/0\.81 and 4\.2/)).toBeTruthy()
  })

  it('groups evidence under its kind, in a fixed order', () => {
    renderAnswer(
      anAnswer({
        evidence: [
          { kind: 'DOCUMENTED', text: 'Attach an accelerometer.', source: 'Bearing SOP v1.4' },
          { kind: 'OBSERVED', text: 'vibration 6.18 mm/s', source: null },
          { kind: 'PREDICTED', text: 'probability 0.998', source: null },
        ],
      }),
    )

    const headings = screen.getAllByRole('heading', { level: 4 }).map((node) => node.textContent)

    expect(headings).toEqual(['Observed', 'Predicted', 'Documented'])
    // A documented claim always shows where it came from -- the domain refuses
    // to construct one without a source, and the page has to render it.
    expect(screen.getByText('Bearing SOP v1.4')).toBeTruthy()
  })

  it('lists the tools that were consulted, one line each', () => {
    // The activity line is bounded: the retrieval tool's summary is the
    // passages themselves, and printing that here is a paragraph where a line
    // belongs. This asserts the rendered line, which is the half the API's
    // tests cannot see.
    renderAnswer(
      anAnswer({
        tool_calls: [
          { tool: 'get_machine_trend', summary: '4 of 6 signals moved', evidence_count: 8 },
          { tool: 'search_maintenance_knowledge', summary: '3 passages from Bearing SOP', evidence_count: 3 },
        ],
      }),
    )

    const consulted = screen.getByRole('heading', { name: /consulted/i }).parentElement
    expect(within(consulted as HTMLElement).getByText(/4 of 6 signals moved/)).toBeTruthy()
    expect(within(consulted as HTMLElement).getByText(/3 passages from Bearing SOP/)).toBeTruthy()
  })

  it('links the answer to the machine it is about', () => {
    renderAnswer(anAnswer())

    expect(screen.getByRole('link', { name: 'M003' }).getAttribute('href')).toBe('/machines/M003')
  })
})

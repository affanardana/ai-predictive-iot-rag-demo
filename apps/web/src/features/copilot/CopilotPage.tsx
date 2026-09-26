/**
 * The Copilot route, which Phase 10 owns.
 *
 * Renders the vocabulary the feature is built on rather than nothing at all,
 * so the page communicates the product's model instead of being blank. The
 * four categories are PRD section 16's, and they are the reason the rest of
 * this dashboard is careful about showing measurement and aggregate
 * differently.
 */

import { Card } from '@/components/Card'
import { PhasePlaceholder } from '@/components/PhasePlaceholder'

const EVIDENCE = [
  {
    name: 'Observed',
    detail: 'A measurement that was actually recorded — a sensor reading, a stored timestamp.',
  },
  {
    name: 'Predicted',
    detail: 'What the model computed. An estimate with a horizon, never a diagnosis.',
  },
  {
    name: 'Documented',
    detail: 'Maintenance procedure retrieved from the manuals, with the source shown.',
  },
  {
    name: 'Inferred',
    detail: 'What the assistant concluded by combining the three above.',
  },
] as const

export function CopilotPage() {
  return (
    <PhasePlaceholder
      phase="Phase 10"
      title="Maintenance Copilot"
      summary="Asking questions about a machine in plain language, and getting an answer that separates what was measured from what was predicted and what the manuals say."
      planned={[
        'Conversation history',
        'Answers combining current state, recent trend, prediction and maintenance documentation',
        'Sources displayed alongside the answer',
        'Tool activity, so a reader can see what was consulted',
      ]}
    >
      <Card
        title="The four kinds of statement"
        subtitle="Distinguishing these is the point of the feature, not a presentation detail."
      >
        <dl className="grid gap-3 sm:grid-cols-2">
          {EVIDENCE.map((item) => (
            <div key={item.name} className="rounded border border-line px-3 py-2">
              <dt className="text-sm font-medium">{item.name}</dt>
              <dd className="mt-0.5 text-sm text-ink-muted">{item.detail}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-sm text-ink-muted">
          An assistant that cannot say which of these it is doing will present a guess about a
          bearing in the same voice it uses for a measured temperature. The retrieval work that
          makes &ldquo;Documented&rdquo; possible is Phase 9.
        </p>
      </Card>
    </PhasePlaceholder>
  )
}

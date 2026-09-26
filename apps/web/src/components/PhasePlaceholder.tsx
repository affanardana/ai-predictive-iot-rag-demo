/**
 * A route that exists but has no backend yet.
 *
 * A real page rather than a redirect or a disabled button. A greyed-out "Start
 * simulation" implies the feature is present and broken, which is a worse
 * impression than an honest statement of what is coming -- and this project
 * already leaves `IncidentType.UNCLASSIFIED` visible in the product rather than
 * hiding it, so saying so plainly is the established convention.
 */

import type { ReactNode } from 'react'

import { Card } from '@/components/Card'

export function PhasePlaceholder({
  phase,
  title,
  summary,
  planned,
  children,
}: {
  phase: string
  title: string
  summary: string
  planned: readonly string[]
  children?: ReactNode
}) {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div>
        <p className="text-sm font-medium text-accent">{phase}</p>
        <h1 className="mt-1 text-2xl font-semibold">{title}</h1>
      </div>

      <Card>
        <p className="text-sm text-ink">{summary}</p>
        <p className="mt-4 text-sm font-medium">Planned for this page</p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-ink-muted">
          {planned.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </Card>

      {children}
    </div>
  )
}

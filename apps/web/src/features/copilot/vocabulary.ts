/**
 * What the four kinds of statement mean, and what each tool is called.
 *
 * One place, because each is used twice: as the key printed beside a finished
 * answer's evidence, and in the explanation of what those kinds are. Written
 * twice, the label and its explanation drift apart, and the feature's whole
 * claim is that a reader can tell a measurement from a guess.
 *
 * Both maps are keyed by a union mirrored from the server in `api/stream.ts`,
 * which makes them the exhaustiveness check for a contract OpenAPI cannot
 * describe: a fifth evidence kind or an eighth tool fails `npm run typecheck`
 * here until it is given a name, rather than rendering a blank heading.
 */

import type { CopilotTool, EvidenceKind } from '@/api/stream'

/** PRD section 16's four vocabularies. */
export const EVIDENCE_VOCABULARY: Record<EvidenceKind, { name: string; detail: string }> = {
  OBSERVED: {
    name: 'Observed',
    detail: 'A measurement that was actually recorded — a sensor reading, a stored timestamp.',
  },
  PREDICTED: {
    name: 'Predicted',
    detail: 'What the model computed. An estimate with a horizon, never a diagnosis.',
  },
  DOCUMENTED: {
    name: 'Documented',
    detail: 'Maintenance procedure retrieved from the manuals, with the source shown.',
  },
  INFERRED: {
    name: 'Inferred',
    detail: 'What the system concluded by combining the three above.',
  },
}

/** The tools `MASTERPLAN.md` section 6 names, as a reader would say them. */
export const TOOL_NAMES: Record<CopilotTool, string> = {
  get_machine_current_state: 'Current state',
  get_machine_telemetry: 'Telemetry',
  get_machine_trend: 'Trend',
  get_machine_prediction: 'Prediction',
  get_machine_incidents: 'Incident history',
  get_machine_history: 'Activity summary',
  search_maintenance_knowledge: 'Maintenance documentation',
}

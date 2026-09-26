/**
 * The corpus, as a table.
 *
 * Every version is listed, with the active one marked. The distinction is the
 * point: PRD section 18 shows a version beside every citation, and the versions
 * that are no longer active are exactly the ones an older citation still points
 * at -- so they are shown rather than filtered out.
 *
 * `is_synthetic` is shown as a column rather than assumed. PRD section 17
 * requires the documentation to be identifiable as demonstration material, and
 * this is where a reader of the product sees that said.
 */

import type { KnowledgeDocument } from '@/api/types'
import { formatInstant } from '@/lib/time'

export function CorpusTable({ documents }: { documents: readonly KnowledgeDocument[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[44rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-ink-muted">
            <th className="py-2 pr-4 font-medium">Document</th>
            <th className="py-2 pr-4 font-medium">Version</th>
            <th className="py-2 pr-4 font-medium">Category</th>
            <th className="py-2 pr-4 text-right font-medium">Pages</th>
            <th className="py-2 pr-4 font-medium">Ingested</th>
            <th className="py-2 font-medium">Retrievable</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((document) => (
            <tr key={document.document_id} className="border-b border-line/60">
              <td className="py-2 pr-4">
                <span className="font-medium">{document.title}</span>
                {document.is_synthetic && (
                  <span className="ml-2 rounded bg-canvas px-1.5 py-0.5 text-xs text-ink-muted">
                    demonstration material
                  </span>
                )}
              </td>
              <td className="py-2 pr-4 font-mono text-xs">{document.version}</td>
              <td className="py-2 pr-4 text-xs text-ink-muted">
                {document.category.replaceAll('_', ' ').toLowerCase()}
              </td>
              <td className="py-2 pr-4 text-right">{document.page_count}</td>
              <td className="py-2 pr-4 text-xs text-ink-muted">
                {formatInstant(document.ingested_at)}
              </td>
              <td className="py-2">
                {document.is_active ? (
                  <span className="text-accent">active</span>
                ) : (
                  <span className="text-ink-muted">withdrawn</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

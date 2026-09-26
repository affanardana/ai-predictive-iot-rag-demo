/**
 * The maintenance corpus, and a way to ask it questions.
 *
 * This is the retrieval layer PRD section 20.6 will sit on, shown on its own so
 * the step between "a question" and "a cited passage" is visible rather than
 * hidden inside the Copilot. Phase 10 builds the answer; this page shows the
 * evidence it will be given.
 *
 * The insufficiency state is the point of the page as much as the results are.
 * PRD section 19 requires the system to say when the documentation does not
 * cover something, and a search box that always returns its nearest neighbour
 * cannot show that -- so "the corpus does not answer this" is rendered as a
 * state, not as an empty list.
 */

import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import { ApiError } from '@/api/errors'
import { knowledgeDocumentsQuery, searchMaintenanceKnowledge } from '@/api/queries'
import type { KnowledgeMatch, SearchKnowledgeResponse } from '@/api/types'
import { Card } from '@/components/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { CorpusTable } from '@/features/knowledge/CorpusTable'

/** The question the demonstration turns on, offered as the starting point. */
const EXAMPLE_QUESTION = 'Vibration is rising on M003. What should I inspect?'

export function KnowledgePage() {
  const [question, setQuestion] = useState(EXAMPLE_QUESTION)
  const documents = useQuery(knowledgeDocumentsQuery())

  const search = useMutation({
    // `rerank` is sent explicitly rather than left off. It is optional on the
    // wire and required in the generated types — a schema with a default is
    // typed as always present, since the server would fill it in anyway — and
    // the page does want the cross-encoder's ordering.
    mutationFn: (query: string) => searchMaintenanceKnowledge({ query, rerank: true }),
  })

  const submitted = question.trim() !== ''

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Knowledge</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Maintenance documentation is retrieved by meaning rather than by keyword, then ordered by
          a cross-encoder. Every passage comes back with the document, version, section and page it
          was taken from.
        </p>
      </div>

      <Card title="Ask the corpus">
        <form
          className="space-y-3"
          onSubmit={(event) => {
            event.preventDefault()
            if (submitted) search.mutate(question.trim())
          }}
        >
          <label className="block text-sm font-medium" htmlFor="question">
            Question
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              id="question"
              className="min-w-64 flex-1 rounded border border-line bg-canvas px-3 py-2 text-sm"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What should be inspected when vibration rises?"
            />
            <button
              type="submit"
              disabled={!submitted || search.isPending}
              className="rounded bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {search.isPending ? 'Searching…' : 'Search'}
            </button>
          </div>
        </form>

        <div className="mt-4">
          {search.isPending && <LoadingState label="Searching the corpus…" />}
          {search.isError && <SearchError error={search.error} />}
          {search.isSuccess && <SearchResults response={search.data} />}
        </div>
      </Card>

      <Card
        title="Corpus"
        subtitle="Every ingested version. Only active versions can be cited."
      >
        {documents.isPending && <LoadingState />}
        {documents.isError && <ErrorState error={documents.error} />}
        {documents.isSuccess &&
          (documents.data.length === 0 ? (
            <EmptyState>
              No documentation has been ingested yet. Run <code>ml knowledge ingest</code> against
              the corpus manifest.
            </EmptyState>
          ) : (
            <CorpusTable documents={documents.data} />
          ))}
      </Card>
    </div>
  )
}

function SearchResults({ response }: { response: SearchKnowledgeResponse }) {
  if (!response.sufficient) {
    return (
      <div className="rounded border border-line bg-canvas px-4 py-3">
        <p className="text-sm font-medium">The documentation does not cover this.</p>
        <p className="mt-1 text-sm text-ink-muted">{response.reason}</p>
        {response.matches.length > 0 && (
          <p className="mt-2 text-xs text-ink-muted">
            Closest passages are shown below for reference; they are not evidence for an answer.
          </p>
        )}
        <ul className="mt-3 space-y-2">
          {response.matches.map((match) => (
            <MatchItem key={match.citation.document_key + match.citation.page + match.content} match={match} muted />
          ))}
        </ul>
      </div>
    )
  }

  return (
    <ol className="space-y-3">
      {response.matches.map((match, index) => (
        <MatchItem
          key={match.citation.document_key + match.citation.page + match.content}
          match={match}
          rank={index + 1}
        />
      ))}
    </ol>
  )
}

function MatchItem({
  match,
  rank,
  muted = false,
}: {
  match: KnowledgeMatch
  rank?: number
  muted?: boolean
}) {
  return (
    <li
      className={`rounded border border-line px-3 py-2 ${muted ? 'opacity-60' : 'bg-surface'}`}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">
          {rank !== undefined && <span className="mr-2 text-ink-muted">{rank}.</span>}
          {match.citation.label}
        </p>
        <span className="font-mono text-xs text-ink-muted">{match.score.toFixed(2)}</span>
      </div>
      <p className="mt-1 whitespace-pre-line text-sm text-ink-muted">{match.content}</p>
    </li>
  )
}

/**
 * A retrieval failure is not "no documentation".
 *
 * The API answers 503 when the embedding or reranking service is unreachable,
 * and that is a statement about the deployment rather than about the corpus --
 * showing it as an empty result would tell a reader the procedure does not
 * exist when it may well.
 */
function SearchError({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.code === 'retrieval_unavailable') {
    return (
      <div className="rounded border border-risk-warning/30 bg-risk-warning/5 px-4 py-3">
        <p className="text-sm font-medium text-risk-warning">
          The retrieval service is not answering.
        </p>
        <p className="mt-1 text-sm text-ink-muted">
          This is a deployment problem, not a missing document. Try again once the inference
          service is up.
        </p>
      </div>
    )
  }
  return <ErrorState error={error} />
}

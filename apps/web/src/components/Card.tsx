import type { ReactNode } from 'react'

/**
 * A titled panel.
 *
 * A component rather than repeated class strings so the page rhythm is changed
 * in one place -- which matters more here than usual, because a dashboard is
 * mostly panels and inconsistent spacing between them is the fastest way to
 * make one look unfinished.
 */
export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`rounded-lg border border-line bg-surface ${className}`}>
      {(title !== undefined || actions !== undefined) && (
        <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line px-4 py-3">
          <div>
            {title !== undefined && <h2 className="text-base font-semibold">{title}</h2>}
            {subtitle !== undefined && (
              <p className="mt-0.5 text-sm text-ink-muted">{subtitle}</p>
            )}
          </div>
          {actions}
        </header>
      )}
      <div className="px-4 py-4">{children}</div>
    </section>
  )
}

/**
 * A section that can be folded away, with the interaction rules in one place.
 *
 * Extracted when the landing page gained its second panel. Every detail below
 * was arrived at by getting it wrong once, and a second hand-rolled copy would
 * have re-derived them slightly differently:
 *
 *   - **The whole header is the click target**, which matters on a touch screen,
 *     while the control on the right is what *looks* clickable. Styling only the
 *     text as a button leaves a small target; styling only the header as one, as
 *     this was, leaves something that reads as a heading because it is rendered
 *     exactly like every other panel title on the page.
 *   - **`cursor-pointer`, which is not decoration.** Tailwind v4's preflight no
 *     longer sets it on buttons, so without this the pointer never changes and
 *     the control feels inert even to someone who has decided to click it.
 *   - **Open by default.** It started collapsed with a bare `+` on the right,
 *     which failed twice over: a reader could not tell it was a control, and the
 *     one person it exists for -- a reviewer arriving cold, who `MASTERPLAN.md`
 *     §27 says must understand the project without the developer explaining it --
 *     was the one who would never open it.
 */

import { useState, type ReactNode } from 'react'

export function CollapsiblePanel({
  title,
  id,
  children,
}: {
  title: string
  /** Anchors the body to its control, for `aria-controls`. */
  id: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(true)

  return (
    <section className="rounded-lg border border-line bg-surface">
      <button
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        aria-expanded={open}
        aria-controls={id}
        className="flex w-full cursor-pointer items-center justify-between gap-3 rounded-lg px-4 py-3 text-left hover:bg-canvas"
      >
        <span className="flex items-center gap-2">
          <span
            aria-hidden="true"
            className={`text-xs text-accent transition-transform ${open ? 'rotate-90' : ''}`}
          >
            ▶
          </span>
          <span className="text-sm font-medium">{title}</span>
        </span>
        <span className="rounded border border-line px-2 py-0.5 text-xs text-ink-muted">
          {open ? 'Hide' : 'Show'}
        </span>
      </button>

      {/* Rendered conditionally rather than hidden with CSS: a reader using a
          screen reader should not have to tab through a folded section. */}
      {open && (
        <div id={id} className="border-t border-line px-4 py-4 text-sm">
          {children}
        </div>
      )}
    </section>
  )
}

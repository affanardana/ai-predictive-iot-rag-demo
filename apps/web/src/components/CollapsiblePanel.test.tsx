import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { CollapsiblePanel } from '@/components/CollapsiblePanel'

/**
 * The behaviour two landing-page panels now share.
 *
 * Tested here rather than through either panel, because the details are the
 * kind that get re-derived slightly differently the second time: the body is
 * *absent* when collapsed rather than hidden, and `aria-expanded` tracks what a
 * reader sees.
 */
function renderPanel() {
  return render(
    <CollapsiblePanel title="How to play" id="how-to-play">
      <p>Step one</p>
    </CollapsiblePanel>,
  )
}

describe('CollapsiblePanel', () => {
  it('is open on arrival', () => {
    // It started collapsed once, with a bare `+` for a control. The reader it
    // exists for -- someone arriving cold -- never opened it.
    renderPanel()

    expect(screen.getByText('Step one')).toBeTruthy()
  })

  it('folds the body away and says so', () => {
    renderPanel()
    const control = screen.getByRole('button', { name: /how to play/i })
    expect(control.getAttribute('aria-expanded')).toBe('true')

    fireEvent.click(control)

    expect(control.getAttribute('aria-expanded')).toBe('false')
    // Absent, not merely hidden: a screen reader should not tab through a
    // folded section.
    expect(screen.queryByText('Step one')).toBeNull()
  })

  it('unfolds again', () => {
    renderPanel()
    const control = screen.getByRole('button', { name: /how to play/i })

    fireEvent.click(control)
    fireEvent.click(control)

    expect(screen.getByText('Step one')).toBeTruthy()
    expect(control.getAttribute('aria-expanded')).toBe('true')
  })
})

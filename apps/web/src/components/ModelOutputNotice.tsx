/**
 * The sentence PRD section 9 requires.
 *
 * "The UI must clearly communicate that a prediction represents model output
 * rather than certainty." A tooltip does not satisfy that -- it is invisible
 * until hovered, absent on touch, and unseen by anyone reading a screenshot.
 *
 * Rendered wherever a probability is presented as a headline number, which is
 * the point at which a reader is most likely to take it for a diagnosis.
 */

export function ModelOutputNotice({ className = '' }: { className?: string }) {
  return (
    <p className={`text-sm text-ink-muted ${className}`}>
      Failure probability is model output, not a certainty and not a diagnosis. The model is a
      binary classifier: it detects that risk is rising, and cannot say what is failing — which is
      why every incident it raises is type <span className="font-medium">Unclassified</span>.
    </p>
  )
}

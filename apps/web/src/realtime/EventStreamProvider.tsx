/**
 * One `EventSource` for the whole application.
 *
 * Mounted at the root rather than per page, for two reasons. Browsers cap
 * concurrent `EventSource` connections per origin at six on HTTP/1.1, so one
 * per component would exhaust that with a handful of panels. And navigating
 * between routes must not tear the connection down and rebuild it, or every
 * page change would show stale data until the next poll.
 */

import { useEffect, useRef, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { API_BASE_URL } from '@/config'
import { everything, invalidationsFor, type QueryKey } from '@/realtime/invalidations'
import { parseMachineEvent, type MachineEvent } from '@/realtime/types'

/**
 * How long events are gathered before the cache is invalidated.
 *
 * The simulator publishes about one reading per second per machine. Without
 * this, each frame would be its own refetch, so a single open dashboard would
 * issue several requests a second against a one-core server -- and a burst of
 * events after a pause would arrive as a burst of requests.
 *
 * Coalescing is safe because an event is a hint: the client refetches
 * authoritative state, so a delayed hint costs freshness rather than
 * correctness, and duplicate hints within one window collapse to one refetch.
 */
const COALESCE_MS = 750

export function EventStreamProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  // A ref rather than state: these are bookkeeping for the timer, and
  // re-rendering the whole tree because a frame arrived would be absurd.
  const pending = useRef<MachineEvent[]>([])
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    const flush = () => {
      timer.current = undefined
      const events = pending.current
      pending.current = []

      // Deduplicated across the window: two frames about the same machine
      // invalidate the same keys, and invalidating twice would refetch twice.
      const seen = new Set<string>()
      for (const event of events) {
        for (const key of invalidationsFor(event)) {
          const identity = JSON.stringify(key)
          if (seen.has(identity)) continue
          seen.add(identity)
          void queryClient.invalidateQueries({ queryKey: key as QueryKey })
        }
      }
    }

    const source = new EventSource(`${API_BASE_URL}/api/v1/events`)

    const onMessage = (message: MessageEvent<string>) => {
      const event = parseMachineEvent(message.data)
      // One unreadable frame is not worth reporting. The next one repairs
      // whatever it was about to say.
      if (event === undefined) return

      pending.current.push(event)
      timer.current ??= window.setTimeout(flush, COALESCE_MS)
    }

    const onOpen = () => {
      // A reconnect may have missed frames, and a first connection has nothing
      // cached yet -- either way the answer is to refetch everything. This is
      // the whole reconnect strategy, and it works only because the stream
      // carries hints rather than state.
      for (const key of everything()) {
        void queryClient.invalidateQueries({ queryKey: key as QueryKey })
      }
    }

    source.addEventListener('change', onMessage)
    source.addEventListener('open', onOpen)

    return () => {
      source.removeEventListener('change', onMessage)
      source.removeEventListener('open', onOpen)
      source.close()
      if (timer.current !== undefined) window.clearTimeout(timer.current)
    }
  }, [queryClient])

  return <>{children}</>
}

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router-dom'

import { EventStreamProvider } from '@/realtime/EventStreamProvider'
import { router } from '@/routes'
import '@/index.css'

/**
 * One cache for the application.
 *
 * `retry: 1` rather than the default three: the API is on the same network as
 * the browser and a failure is usually real -- a 404 for a machine that is not
 * registered, or a 503 while the inference service reloads. Retrying those
 * three times with backoff makes the dashboard feel broken rather than
 * informative, and the query definitions already poll, so the next poll is the
 * retry.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      // The stream invalidates what it needs to, and every query declares its
      // own poll interval. A default `staleTime` of zero would refetch on
      // every mount, including a navigation back to a page just left.
      staleTime: 5_000,
      refetchOnWindowFocus: false,
    },
  },
})

const container = document.getElementById('root')
if (container === null) throw new Error('index.html has no #root element.')

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      {/* Inside the query provider, because it invalidates queries. Outside
          the router, so navigating between pages does not reconnect. */}
      <EventStreamProvider>
        <RouterProvider router={router} />
      </EventStreamProvider>
    </QueryClientProvider>
  </StrictMode>,
)

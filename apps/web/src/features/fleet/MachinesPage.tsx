/**
 * The machine list, on its own page.
 *
 * The same table the overview shows. Kept separate because the routes are
 * specified separately and because this is the page a link to "all machines"
 * should land on as the fleet grows past what an overview can hold.
 */

import { useQuery } from '@tanstack/react-query'

import { fleetQuery } from '@/api/queries'
import { Card } from '@/components/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { MachineTable } from '@/features/fleet/MachineTable'

export function MachinesPage() {
  const fleet = useQuery(fleetQuery())

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Machines</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Select a machine to see its telemetry, prediction history and incidents.
        </p>
      </div>

      <Card>
        {fleet.isPending && <LoadingState label="Loading machines…" />}
        {fleet.isError && <ErrorState error={fleet.error} />}
        {fleet.data !== undefined &&
          (fleet.data.length === 0 ? (
            <EmptyState>No machines are registered yet.</EmptyState>
          ) : (
            <MachineTable summaries={fleet.data} />
          ))}
      </Card>
    </div>
  )
}

/**
 * The landing page: what the fleet is doing, at a glance.
 */

import { useQuery } from '@tanstack/react-query'

import { fleetQuery } from '@/api/queries'
import { Card } from '@/components/Card'
import { EmptyState, ErrorState, LoadingState } from '@/components/States'
import { FleetSummary } from '@/features/fleet/FleetSummary'
import { MachineTable } from '@/features/fleet/MachineTable'
import { AboutPanel } from '@/features/fleet/AboutPanel'

export function OverviewPage() {
  const fleet = useQuery(fleetQuery())

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Fleet overview</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Simulated industrial motors, scored continuously from their telemetry.
        </p>
      </div>

      <AboutPanel />

      {fleet.isPending && <LoadingState label="Loading the fleet…" />}
      {fleet.isError && <ErrorState error={fleet.error} />}

      {fleet.data !== undefined && (
        <>
          {fleet.data.length === 0 ? (
            <Card>
              <EmptyState>
                No machines are registered yet. A machine must be registered before the pipeline
                will accept its telemetry. The command is in{' '}
                <code className="font-mono text-xs">infra/compose/README.md</code>, under
                &ldquo;Running the demonstration&rdquo;.
              </EmptyState>
            </Card>
          ) : (
            <>
              <Card>
                <FleetSummary summaries={fleet.data} />
              </Card>
              <Card
                title="Machines"
                subtitle="Every registered machine, with its latest reading and prediction."
              >
                <MachineTable summaries={fleet.data} />
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}

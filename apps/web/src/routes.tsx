/**
 * The route table.
 *
 * Six routes, matching PRD section 20's list exactly. Two of them have no
 * backend yet and say so.
 */

import { createBrowserRouter } from 'react-router-dom'

import { AppShell } from '@/components/AppShell'
import { EmptyState } from '@/components/States'
import { CopilotPage } from '@/features/copilot/CopilotPage'
import { IncidentsPage } from '@/features/incidents/IncidentsPage'
import { MachinePage } from '@/features/machine/MachinePage'
import { MachinesPage } from '@/features/fleet/MachinesPage'
import { OverviewPage } from '@/features/fleet/OverviewPage'
import { SimulationPage } from '@/features/simulation/SimulationPage'

function NotFoundPage() {
  return <EmptyState>That page does not exist.</EmptyState>
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: 'machines', element: <MachinesPage /> },
      { path: 'machines/:machineId', element: <MachinePage /> },
      { path: 'incidents', element: <IncidentsPage /> },
      { path: 'simulation', element: <SimulationPage /> },
      { path: 'copilot', element: <CopilotPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])

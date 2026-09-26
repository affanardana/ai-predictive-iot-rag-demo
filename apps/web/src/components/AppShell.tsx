/**
 * The frame every page sits in.
 */

import { NavLink, Outlet } from 'react-router-dom'

import { API_DOCS_URL } from '@/config'

const NAVIGATION = [
  { to: '/', label: 'Overview', end: true },
  { to: '/machines', label: 'Machines' },
  { to: '/incidents', label: 'Incidents' },
  { to: '/simulation', label: 'Simulation' },
  { to: '/knowledge', label: 'Knowledge' },
  { to: '/copilot', label: 'Copilot' },
] as const

export function AppShell() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <NavLink to="/" className="text-base font-semibold tracking-tight">
            Predictive Maintenance
          </NavLink>

          <nav className="flex flex-wrap gap-x-1 gap-y-1">
            {NAVIGATION.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={'end' in item ? item.end : false}
                className={({ isActive }) =>
                  `rounded px-2.5 py-1 text-sm ${
                    isActive
                      ? 'bg-accent/10 font-medium text-accent'
                      : 'text-ink-muted hover:bg-canvas hover:text-ink'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <a
            href={API_DOCS_URL}
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-sm text-ink-muted hover:text-ink"
          >
            API reference ↗
          </a>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 pt-2 text-xs text-ink-muted">
        Simulated fleet. Risk bands are product configuration for this demonstration, not
        industrial thresholds.
      </footer>
    </div>
  )
}

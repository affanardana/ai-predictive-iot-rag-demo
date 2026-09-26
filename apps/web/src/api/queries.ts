/**
 * Every query key, and every fetch, in one module.
 *
 * **Nothing outside this file may build a query key.** Invalidation works by
 * prefix match, so a key assembled anywhere else is a key that can silently
 * stop matching -- and the symptom is a dashboard that looks frozen rather
 * than an error anyone can trace. One builder, one place to look.
 */

import { queryOptions } from '@tanstack/react-query'

import { deleteJson, getJson, patchJson, postEmpty, postJson } from '@/api/client'
import type {
  Incident,
  IncidentStatus,
  MachineDetail,
  MachineSummary,
  Prediction,
  SimulationRun,
  StartSimulationRequest,
  TelemetrySeries,
  TimeWindow,
} from '@/api/types'

/**
 * Cache keys, built rather than written.
 *
 * The nesting is the invalidation grammar: `['machines']` matches every fleet
 * query, `['machine', id]` matches everything about one machine, and
 * `['machine', id, 'telemetry', window]` matches one chart. Invalidating a
 * prefix is how a single event refreshes exactly what it affects.
 */
export const keys = {
  all: ['machines'] as const,
  fleet: () => [...keys.all] as const,
  machine: (machineId: string) => ['machine', machineId] as const,
  machineDetail: (machineId: string) => [...keys.machine(machineId), 'detail'] as const,
  telemetry: (machineId: string, window: TimeWindow) =>
    [...keys.machine(machineId), 'telemetry', window] as const,
  /** Every window for one machine, for invalidation when a reading arrives. */
  telemetryAll: (machineId: string) => [...keys.machine(machineId), 'telemetry'] as const,
  predictions: (machineId: string) => [...keys.machine(machineId), 'predictions'] as const,
  machineIncidents: (machineId: string) => [...keys.machine(machineId), 'incidents'] as const,
  incidents: ['incidents'] as const,
  incidentList: (status?: IncidentStatus) => [...keys.incidents, status ?? 'ALL'] as const,
  simulations: () => ['simulations'] as const,
  machineSimulations: (machineId: string) =>
    [...keys.machine(machineId), 'simulations'] as const,
}

/**
 * How often a page refetches even when nothing is announced.
 *
 * The event stream is a latency optimisation, never the source of correctness.
 * A dropped frame, a proxy that eats the connection, or a browser tab that was
 * asleep all degrade to "stale until the next poll" rather than to "wrong
 * forever" -- which is what makes it safe for the stream to drop events and
 * for the broadcaster not to replay them.
 */
const POLL_INTERVAL_MS = 30_000

export const fleetQuery = () =>
  queryOptions({
    queryKey: keys.fleet(),
    queryFn: ({ signal }) => getJson<MachineSummary[]>('/api/v1/machines', undefined, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })

export const machineDetailQuery = (machineId: string) =>
  queryOptions({
    queryKey: keys.machineDetail(machineId),
    queryFn: ({ signal }) =>
      getJson<MachineDetail>(`/api/v1/machines/${machineId}`, undefined, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })

export const telemetryQuery = (machineId: string, window: TimeWindow) =>
  queryOptions({
    queryKey: keys.telemetry(machineId, window),
    queryFn: ({ signal }) =>
      getJson<TelemetrySeries>(`/api/v1/machines/${machineId}/telemetry`, { window }, signal),
    // Historical windows change slowly; a minute is responsive enough and
    // keeps the 30-day aggregation off the wire.
    refetchInterval: 60_000,
  })

export const predictionsQuery = (machineId: string) =>
  queryOptions({
    queryKey: keys.predictions(machineId),
    queryFn: ({ signal }) =>
      getJson<Prediction[]>(`/api/v1/machines/${machineId}/predictions`, undefined, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })

export const machineIncidentsQuery = (machineId: string) =>
  queryOptions({
    queryKey: keys.machineIncidents(machineId),
    queryFn: ({ signal }) =>
      getJson<Incident[]>(`/api/v1/machines/${machineId}/incidents`, undefined, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })

export const incidentListQuery = (status?: IncidentStatus) =>
  queryOptions({
    queryKey: keys.incidentList(status),
    queryFn: ({ signal }) =>
      getJson<Incident[]>('/api/v1/incidents', status === undefined ? {} : { status }, signal),
    refetchInterval: POLL_INTERVAL_MS,
  })

/**
 * How often an *active* run's progress is polled.
 *
 * Far quicker than the 30-second floor everything else uses, because a progress
 * bar that moves twice a minute is not a progress bar. Progress deliberately
 * does not travel on the event stream -- the simulator reports every few
 * seconds, and announcing each report would make every open dashboard refetch
 * on that cadence for a number nobody is watching that closely.
 */
const ACTIVE_RUN_POLL_MS = 3_000

export const simulationsQuery = () =>
  queryOptions({
    queryKey: keys.simulations(),
    queryFn: ({ signal }) => getJson<SimulationRun[]>('/api/v1/simulations', undefined, signal),
    // A function so the cadence depends on whether anything is actually
    // running: an idle fleet polls at the ordinary floor, and only a live run
    // asks for updates this often.
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.is_active) ? ACTIVE_RUN_POLL_MS : POLL_INTERVAL_MS,
  })

export const machineSimulationsQuery = (machineId: string) =>
  queryOptions({
    queryKey: keys.machineSimulations(machineId),
    queryFn: ({ signal }) =>
      getJson<SimulationRun[]>(`/api/v1/machines/${machineId}/simulations`, undefined, signal),
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.is_active) ? ACTIVE_RUN_POLL_MS : POLL_INTERVAL_MS,
  })

/** Begin a run. */
export function startSimulation(body: StartSimulationRequest): Promise<SimulationRun> {
  return postJson<SimulationRun>('/api/v1/simulations', body)
}

/** End a run, keeping its record. */
export function stopSimulation(sessionId: string): Promise<SimulationRun> {
  return postEmpty<SimulationRun>(`/api/v1/simulations/${sessionId}/stop`)
}

/** Discard a finished run's record, freeing the machine for another. */
export function resetSimulation(sessionId: string): Promise<void> {
  return deleteJson(`/api/v1/simulations/${sessionId}`)
}

/**
 * Move an incident to a new status.
 *
 * The target is a status rather than an action, mirroring the API and the
 * domain behind it: the permitted moves are a table over statuses, and
 * inventing verb-shaped client calls would put a second copy of that table
 * here.
 */
export function updateIncidentStatus(incidentId: string, status: IncidentStatus): Promise<Incident> {
  return patchJson<Incident>(`/api/v1/incidents/${incidentId}`, { status })
}

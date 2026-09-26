/**
 * Names for the API's schemas.
 *
 * `schema.d.ts` is generated and its types are reached through
 * `components['schemas'][...]`, which is accurate and unreadable. This module
 * gives each one a name, in one place, so a schema rename breaks here rather
 * than in twenty components.
 *
 * **`schema.d.ts` does not exist until `npm run gen:api` has been run** -- it
 * is generated from the API's OpenAPI document, which is itself generated from
 * the running code. See `README.md`.
 */

import type { components } from '@/api/schema'

type Schemas = components['schemas']

// --- Fleet and machines ----------------------------------------------------

export type Machine = Schemas['MachineSchema']
export type MachineSummary = Schemas['MachineSummarySchema']
export type MachineDetail = Schemas['MachineDetailSchema']

// --- Telemetry -------------------------------------------------------------

export type SensorReading = Schemas['SensorReadingSchema']
export type TelemetryRecord = Schemas['TelemetryRecordSchema']
export type TelemetryPoint = Schemas['TelemetryPointSchema']
export type TelemetrySeries = Schemas['TelemetrySeriesSchema']
export type SeriesResolution = Schemas['SeriesResolutionSchema']
export type ResolvedWindow = Schemas['ResolvedWindowSchema']

// --- Predictions and incidents ---------------------------------------------

export type Prediction = Schemas['PredictionSchema']
export type Incident = Schemas['IncidentSchema']
export type IncidentStatus = Schemas['IncidentStatus']
export type IncidentSeverity = Schemas['IncidentSeverity']

// --- Simulation ------------------------------------------------------------

export type SimulationRun = Schemas['SimulationRunSchema']
export type SimulationScenario = Schemas['SimulationScenario']
export type RunStatus = Schemas['RunStatus']
export type StartSimulationRequest = Schemas['StartSimulationRequest']

// --- Enums the UI switches on ----------------------------------------------

export type RiskLevel = Schemas['RiskLevel']
export type TimeWindow = Schemas['TimeWindow']
export type Aggregation = Schemas['Aggregation']

/**
 * The six signals, in the order the API reports them.
 *
 * A `const` tuple rather than a type union, because the UI iterates it to draw
 * one panel per signal. Declared here beside the schema types so adding a
 * seventh signal to `SensorReadingSchema` is a change in one file.
 */
export const SENSOR_SIGNALS = [
  'temperature',
  'vibration',
  'rpm',
  'current',
  'load',
  'voltage',
] as const

export type SensorSignal = (typeof SENSOR_SIGNALS)[number]

/** Units, for axis labels. Not in the API -- it reports numbers, not physics. */
export const SIGNAL_UNITS: Record<SensorSignal, string> = {
  temperature: '°C',
  vibration: 'mm/s',
  rpm: 'rpm',
  current: 'A',
  load: 'load',
  voltage: 'V',
}

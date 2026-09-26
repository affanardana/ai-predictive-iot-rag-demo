/**
 * A minimal ECharts binding.
 *
 * Hand-written rather than using `echarts-for-react`: this is about forty
 * lines, it has no dependency to keep in step, and it disposes on unmount
 * without the StrictMode double-mount surprises a wrapper library tends to
 * bring.
 *
 * ECharts is imported **by module** rather than as one bundle. `import * as
 * echarts from 'echarts'` would register every chart type and component the
 * library ships; this application draws time-series lines and nothing else.
 *
 * That keeps the library's contribution proportionate, not small: ECharts is
 * still the largest dependency here, and it is the reason the build reports a
 * chunk-size warning. If the bundle ever needs to shrink, the next lever is
 * `build.rollupOptions.output.manualChunks` to split it away from the
 * application code -- not trimming this list, which would save little.
 */

import { useEffect, useRef } from 'react'

import { LineChart } from 'echarts/charts'
import {
  AxisPointerComponent,
  GridComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  AxisPointerComponent,
  TitleComponent,
  CanvasRenderer,
])

export type { EChartsOption }

/**
 * Render `option` into a div, and keep it sized to its container.
 *
 * Returns the ref to attach; the caller sets the height, because that is a
 * layout decision rather than a chart one.
 */
export function useECharts(option: EChartsOption) {
  const container = useRef<HTMLDivElement | null>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    const element = container.current
    if (element === null) return

    const instance = echarts.init(element)
    chart.current = instance

    // ECharts measures its container once at init. Without this a window
    // resize leaves the canvas at its old width, which looks like a rendering
    // bug rather than a missing listener.
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(element)

    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    // `notMerge: false` so a new option merges into the existing chart rather
    // than rebuilding it -- which is what keeps the live updates from
    // flickering as frames arrive.
    chart.current?.setOption(option, { notMerge: false })
  }, [option])

  return container
}

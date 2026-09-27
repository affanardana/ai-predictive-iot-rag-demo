/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
    react(),
    // Tailwind v4 is a Vite plugin and nothing else. There is no
    // `tailwind.config.js` and no PostCSS config -- the v3 idiom of
    // `@tailwind base; @tailwind components; @tailwind utilities;` produces no
    // styles at all under v4, silently, and looks like a Vite problem. See
    // `src/index.css`.
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // Vitest is pinned to v3 rather than v2 for a reason worth writing down:
  // vitest 2 depends on vite 5, so it installs a second, nested copy of vite
  // under `node_modules/vitest/`. `vitest/config`'s module augmentation then
  // lands on that nested copy and never reaches the vite this file imports --
  // so `test:` is reported as not existing on `UserConfigExport`. Importing
  // `defineConfig` from `vitest/config` instead makes it worse: this config's
  // plugins are vite 6 plugins and fail against vite 5's `PluginOption`.
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.{ts,tsx}'],
    // Rendering *is* tested here now, in a small way: `AnswerBlock.test.tsx`
    // asserts what a reader sees for each of the Copilot's three verdicts.
    //
    // This comment used to say Playwright owned that, in Phase 11. Phase 11
    // declined Playwright -- a browser suite against a live deployment is
    // flaky and slow, and the owner judged it not worth the runner minutes --
    // so rendering is covered where it can be, in jsdom, and the gaps that
    // leaves (routing, styling, ECharts) are real rather than papered over.
    // Still no snapshotting: a snapshot asserts that markup changed, not that
    // it is right.
  },
})

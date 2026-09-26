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
    // Rendering is not tested here. Playwright owns that, in Phase 11 -- so
    // the DOM environment exists for the few component tests that assert what
    // a user can read, not for snapshotting markup.
  },
})

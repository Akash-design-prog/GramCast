import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // maplibre-gl ships its own Web Worker; Vite's dependency pre-bundling breaks the worker's
  // relative URL resolution, so exclude it from that step (see vitejs/vite#8427-style issues).
  optimizeDeps: { exclude: ['maplibre-gl'] },
})

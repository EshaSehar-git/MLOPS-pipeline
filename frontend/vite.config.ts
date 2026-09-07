import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // In dev, the FastAPI backend runs separately on :8000 — proxy API
    // calls there so the same relative fetch("/extract") works in both
    // dev (proxied) and prod (same-origin, since FastAPI serves the build).
    proxy: {
      '/extract': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/model': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})

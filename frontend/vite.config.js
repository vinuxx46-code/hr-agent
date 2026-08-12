import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Browser code uses relative /api URLs. This proxy makes development and hosted
// previews work without hard-coding a candidate's localhost address.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/recordings': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})

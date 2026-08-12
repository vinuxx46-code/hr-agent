import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Browser code uses relative /api URLs. This proxy makes development and hosted
// previews work without hard-coding a candidate's localhost address.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    // Hosted sandbox previews are served from a *.e2b.app subdomain, which
    // Vite blocks by default as a DNS-rebinding protection.
    allowedHosts: ['.e2b.app', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/recordings': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})

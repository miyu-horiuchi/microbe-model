import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// During dev, proxy /api to the FastAPI server on :7860 so we can use one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:7860',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});

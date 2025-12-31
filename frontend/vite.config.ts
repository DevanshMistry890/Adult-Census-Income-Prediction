import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/Adult-Census-Income-Prediction/',
  
  optimizeDeps: {
    exclude: ['webshap', 'node-fetch', 'onnxruntime-web'], 
  },

  build: {
    rollupOptions: {
      external: ['node-fetch'], 
    },
  },

  define: {
    'global': 'window',
  },

  // --- ADD THIS SERVER BLOCK ---
  // This enables the security headers required for Threaded WASM
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    }
  }
});

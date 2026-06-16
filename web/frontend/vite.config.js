import { defineConfig } from 'vite';

export default defineConfig({
  base: '/neuralsat/',
  root: '.',
  build: {
    outDir: '../../docs',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
});

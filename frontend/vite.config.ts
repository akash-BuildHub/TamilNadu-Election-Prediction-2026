import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const devHost = process.env.VITE_DEV_HOST || "0.0.0.0";
const rawDevPort = Number.parseInt(process.env.VITE_DEV_PORT || "5173", 10);
const devPort = Number.isNaN(rawDevPort) ? 5173 : rawDevPort;

export default defineConfig({
  plugins: [react()],
  server: {
    port: devPort,
    host: devHost
  }
});

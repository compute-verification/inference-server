import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so the built bundle works when served from any path (surge).
export default defineConfig({
  base: "./",
  plugins: [react()],
});

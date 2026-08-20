import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  // Keep localhost at / while publishing production assets beneath the
  // GitHub project-page path.
  base: command === "build" ? "/gridworld-agent/" : "/",
}));

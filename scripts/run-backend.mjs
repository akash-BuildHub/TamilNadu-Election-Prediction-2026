#!/usr/bin/env node
// Runs backend/server.py using the project-root .venv interpreter.
// Using the venv's python directly is equivalent to activating the venv.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const IS_WIN = process.platform === "win32";
const VENV_PY = join(ROOT, ".venv", IS_WIN ? "Scripts" : "bin", IS_WIN ? "python.exe" : "python");
const BACKEND_DIR = join(ROOT, "backend");
const SERVER = join(BACKEND_DIR, "server.py");

if (!existsSync(VENV_PY)) {
  console.error(`[backend] ERROR: venv python not found at ${VENV_PY}. Run "npm run setup".`);
  process.exit(1);
}
if (!existsSync(SERVER)) {
  console.error(`[backend] ERROR: server entrypoint missing at ${SERVER}`);
  process.exit(1);
}

const child = spawn(VENV_PY, ["-u", SERVER], {
  cwd: BACKEND_DIR,
  stdio: "inherit",
  env: { ...process.env, PYTHONUNBUFFERED: "1", VIRTUAL_ENV: join(ROOT, ".venv") }
});

const forward = (sig) => () => { if (!child.killed) child.kill(sig); };
process.on("SIGINT", forward("SIGINT"));
process.on("SIGTERM", forward("SIGTERM"));
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 0);
});

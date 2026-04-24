#!/usr/bin/env node
// Creates a project-root .venv and installs requirements.txt on first run.
// Re-runs the install step when requirements.txt changes (hashed in a marker).
// Pass --force to reinstall regardless.

import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const VENV = join(ROOT, ".venv");
const REQS = join(ROOT, "requirements.txt");
const IS_WIN = process.platform === "win32";
const VENV_PY = join(VENV, IS_WIN ? "Scripts" : "bin", IS_WIN ? "python.exe" : "python");
const MARKER = join(VENV, ".setup-complete");
const FORCE = process.argv.includes("--force");

function log(msg) {
  process.stdout.write(`[setup] ${msg}\n`);
}

function findSystemPython() {
  const candidates = IS_WIN ? ["py", "python", "python3"] : ["python3", "python"];
  for (const bin of candidates) {
    const args = bin === "py" ? ["-3", "--version"] : ["--version"];
    const res = spawnSync(bin, args, { stdio: "ignore" });
    if (res.status === 0) return { bin, prefixArgs: bin === "py" ? ["-3"] : [] };
  }
  return null;
}

function createVenv() {
  const py = findSystemPython();
  if (!py) {
    console.error("[setup] ERROR: no system Python found on PATH (tried py, python, python3).");
    process.exit(1);
  }
  log(`creating virtualenv at .venv using ${py.bin} ${py.prefixArgs.join(" ")}`);
  const res = spawnSync(py.bin, [...py.prefixArgs, "-m", "venv", VENV], { stdio: "inherit" });
  if (res.status !== 0) {
    console.error("[setup] ERROR: failed to create .venv");
    process.exit(res.status ?? 1);
  }
}

function reqsHash() {
  return createHash("sha256").update(readFileSync(REQS)).digest("hex");
}

function readMarker() {
  if (!existsSync(MARKER)) return null;
  try {
    return JSON.parse(readFileSync(MARKER, "utf8"));
  } catch {
    return null;
  }
}

function writeMarker(hash) {
  writeFileSync(MARKER, JSON.stringify({ hash, installedAt: new Date().toISOString() }, null, 2));
}

function pipInstall() {
  log("upgrading pip");
  const upgrade = spawnSync(VENV_PY, ["-m", "pip", "install", "--upgrade", "pip"], { stdio: "inherit" });
  if (upgrade.status !== 0) process.exit(upgrade.status ?? 1);
  log("installing requirements.txt (this can take a while on first run — torch is large)");
  const install = spawnSync(VENV_PY, ["-m", "pip", "install", "-r", REQS], { stdio: "inherit" });
  if (install.status !== 0) {
    console.error("[setup] ERROR: pip install failed");
    process.exit(install.status ?? 1);
  }
}

function main() {
  if (!existsSync(REQS)) {
    console.error(`[setup] ERROR: requirements.txt not found at ${REQS}`);
    process.exit(1);
  }

  if (!existsSync(VENV_PY)) {
    createVenv();
  }

  const currentHash = reqsHash();
  const marker = readMarker();
  const needsInstall = FORCE || !marker || marker.hash !== currentHash;

  if (needsInstall) {
    pipInstall();
    writeMarker(currentHash);
    log("venv ready");
  } else {
    log("venv up to date");
  }
}

main();

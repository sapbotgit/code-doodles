#!/usr/bin/env node

const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

/**
 * Parse GPU usage percentage from `rocm-smi -u` output.
 * Expected format: "GPU[0] : GPU use (%): 0" or similar.
 * Returns number or null if not found.
 */
function parseGpuUsage(output) {
  const match = output.match(/GPU use\s*\(\%\):\s*(\d+)/i);
  if (match && match[1]) {
    return parseInt(match[1], 10);
  }
  return null;
}

/**
 * Set GPU fan speed to a given percentage (0-100).
 * Uses `rocm-smi --setfan <percent>%`.
 */
async function setFanSpeed(percent) {
  if (percent < 0 || percent > 100) {
    console.error(`Invalid fan speed: ${percent} (must be 0-100)`);
    return;
  }
  const cmd = `rocm-smi --setfan ${percent}%`;
  try {
    const { stdout, stderr } = await execPromise(cmd);
    if (stderr && !stderr.includes('WARNING')) {
      console.warn('Fan set stderr:', stderr.trim());
    }
    console.log(`Fan set to ${percent}% — ${stdout.match(/Successfully set fan speed/i) ? 'success' : 'check output'}`);
  } catch (error) {
    console.error(`Failed to set fan speed to ${percent}%:`, error.message);
  }
}

/**
 * Fetch current GPU usage and apply as fan speed.
 */
async function updateFanByUsage() {
  try {
    const { stdout, stderr } = await execPromise('rocm-smi -u');
    if (stderr && !stderr.includes('WARNING')) {
      console.warn('GPU usage stderr:', stderr.trim());
    }

    const usagePercent = parseGpuUsage(stdout);
    if (usagePercent === null) {
      console.error('Could not parse GPU usage from rocm-smi output.');
      return;
    }

    console.log(`Current GPU usage: ${usagePercent}%`);
    await setFanSpeed(usagePercent);
  } catch (error) {
    console.error('Error executing rocm-smi -u:', error.message);
  }
}

// Run immediately, then every second
updateFanByUsage();
setInterval(updateFanByUsage, 1000);

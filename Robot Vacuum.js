const rl = require('raylib');
const readline = require('readline');
const romai = require('romai');

// --- Configuration ---
const ROOM_SIZE = 10;
const SCALE = 60;
const WIN_SIZE = ROOM_SIZE * SCALE;
const ROBOT_SIZE = 0.3;
const MOVE_STEP = 0.5;
const GRID_RES = 0.5;
const GRID_CELLS = ROOM_SIZE / GRID_RES;

// --- CLI Parsing ---
const args = process.argv.slice(2);
const isBot = args.includes('--bot');
const aiIndex = args.indexOf('--ai');
let aiModel = null;

if (aiIndex !== -1) {
  if (aiIndex + 1 < args.length && !args[aiIndex + 1].startsWith('-')) {
    aiModel = args[aiIndex + 1];
  } else {
    aiModel = undefined;
  }
}

if (isBot) {
  console.log(`🤖 BOT MODE${aiModel !== null ? ` (AI: ${aiModel || 'default'})` : ` (Algorithm: Perimeter→Snake)`}`);
  console.log('Close window to stop.\n');
}

// --- State ---
const state = {
  robot: { x: 2, y: 2, angle: 0 },
  chair: { x: 6, y: 6, width: 0.6, height: 0.6 },
  distances: { Forward: 0, Left: 0, Right: 0 },
  moveCount: 0,
  cleaned: Array(GRID_CELLS).fill().map(() => Array(GRID_CELLS).fill(false)),
  completed: false,
  phase: 'PERIMETER',
  perimeterStart: null,
  snakeRow: 0,
  snakeDirection: 1,
  targetY: 0.5
};

// Randomize chair
state.chair.x = 4 + Math.random() * 4;
state.chair.y = 4 + Math.random() * 4;

// --- Math Helpers ---
function toRad(deg) { return deg * Math.PI / 180; }
function getVector(angleDeg) {
  const rad = toRad(angleDeg);
  return { x: Math.cos(rad), y: Math.sin(rad) };
}

function getGridCoord(x, y) {
  return {
    x: Math.floor(x / GRID_RES),
    y: Math.floor(y / GRID_RES)
  };
}

// --- Sensors & Physics ---
function rayIntersect(origin, dir, rect) {
  const minX = rect.x - rect.width/2;
  const maxX = rect.x + rect.width/2;
  const minY = rect.y - rect.height/2;
  const maxY = rect.y + rect.height/2;
  let tMin = -Infinity, tMax = Infinity;

  if (dir.x !== 0) {
    const tx1 = (minX - origin.x) / dir.x;
    const tx2 = (maxX - origin.x) / dir.x;
    tMin = Math.max(tMin, Math.min(tx1, tx2));
    tMax = Math.min(tMax, Math.max(tx1, tx2));
  } else if (origin.x < minX || origin.x > maxX) return Infinity;

  if (dir.y !== 0) {
    const ty1 = (minY - origin.y) / dir.y;
    const ty2 = (maxY - origin.y) / dir.y;
    tMin = Math.max(tMin, Math.min(ty1, ty2));
    tMax = Math.min(tMax, Math.max(ty1, ty2));
  } else if (origin.y < minY || origin.y > maxY) return Infinity;

  if (tMax < 0 || tMin > tMax) return Infinity;
  return tMin > 0 ? tMin : Infinity;
}

function wallDistance(origin, dir) {
  let dist = Infinity;
  if (dir.x > 0) dist = Math.min(dist, (ROOM_SIZE - origin.x) / dir.x);
  else if (dir.x < 0) dist = Math.min(dist, -origin.x / dir.x);
  if (dir.y > 0) dist = Math.min(dist, (ROOM_SIZE - origin.y) / dir.y);
  else if (dir.y < 0) dist = Math.min(dist, -origin.y / dir.y);
  return dist;
}

function updateSensors() {
  [
    { key: 'Forward', angle: state.robot.angle },
    { key: 'Left', angle: state.robot.angle - 90 },
    { key: 'Right', angle: state.robot.angle + 90 }
  ].forEach(d => {
    const vec = getVector(d.angle);
    const dChair = rayIntersect(state.robot, vec, state.chair);
    const dWall = wallDistance(state.robot, vec);
    state.distances[d.key] = Math.min(dChair, dWall);
  });
}

function checkCollision(x, y) {
  const r = ROBOT_SIZE / 2;
  if (x - r < 0 || x + r > ROOM_SIZE || y - r < 0 || y + r > ROOM_SIZE) return true;
  const cx = state.chair.x - state.chair.width/2 - r;
  const cy = state.chair.y - state.chair.height/2 - r;
  const cw = state.chair.width + r*2;
  const ch = state.chair.height + r*2;
  if (x > cx && x < cx + cw && y > cy && y < cy + ch) return true;
  return false;
}

function markCleaned() {
  const r = ROBOT_SIZE / 2;
  // Mark all grid cells that the robot physically overlaps
  const minX = Math.floor((state.robot.x - r) / GRID_RES);
  const maxX = Math.floor((state.robot.x + r) / GRID_RES);
  const minY = Math.floor((state.robot.y - r) / GRID_RES);
  const maxY = Math.floor((state.robot.y + r) / GRID_RES);
  
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (y >= 0 && y < GRID_CELLS && x >= 0 && x < GRID_CELLS) {
        state.cleaned[y][x] = true;
      }
    }
  }
}

function getCoverageStats() {
  let cleaned = 0;
  for (let row of state.cleaned) {
    for (let cell of row) if (cell) cleaned++;
  }
  const total = GRID_CELLS * GRID_CELLS;
  return { cleaned, total, percent: (cleaned / total * 100).toFixed(1) };
}

// --- Command Execution ---
function executeCommand(cmd) {
  if (cmd === 'right') {
    state.robot.angle = (state.robot.angle + 90) % 360;
    return true;
  } else if (cmd === 'left') {
    state.robot.angle = (state.robot.angle - 90 + 360) % 360;
    return true;
  } else if (cmd === 'forward') {
    const vec = getVector(state.robot.angle);
    const newX = state.robot.x + vec.x * MOVE_STEP;
    const newY = state.robot.y + vec.y * MOVE_STEP;
    
    if (!checkCollision(newX, newY)) {
      state.robot.x = newX;
      state.robot.y = newY;
      state.moveCount++;
      markCleaned();
      return true;
    } else {
      console.log('*** CLUNK! ***');
      return false;
    }
  } else if (cmd === 'pass') {
    return true;
  }
  return false;
}

// --- Standard Vacuum Algorithm ---
function getAlgorithmicCommand() {
  const canMove = (angle) => {
    const vec = getVector(angle);
    const newX = state.robot.x + vec.x * MOVE_STEP;
    const newY = state.robot.y + vec.y * MOVE_STEP;
    return !checkCollision(newX, newY);
  };

  const getDist = (angle) => {
    const rel = (angle - state.robot.angle + 360) % 360;
    if (rel === 0) return state.distances.Forward;
    if (rel === 90 || rel === -270) return state.distances.Left;
    if (rel === 270 || rel === -90) return state.distances.Right;
    if (rel === 180) {
      const vec = getVector(angle);
      const dChair = rayIntersect(state.robot, vec, state.chair);
      const dWall = wallDistance(state.robot, vec);
      return Math.min(dChair, dWall);
    }
    return 0;
  };  // PHASE 1: PERIMETER FOLLOWING (Fixed)
  if (state.phase === 'PERIMETER') {
    // Initialize with rotation tracking
    if (!state.perimeterStart) {
      state.perimeterStart = { 
        x: state.robot.x, 
        y: state.robot.y,
        startAngle: state.robot.angle,
        totalRotation: 0,
        lastAngle: state.robot.angle,
        hugging: false,
        moves: 0
      };
      state.hadWallOnRight = false;
    }
    
    // Track net rotation (handles 0/360 wrap)
    let angleDelta = state.robot.angle - state.perimeterStart.lastAngle;
    if (angleDelta > 180) angleDelta -= 360;
    if (angleDelta < -180) angleDelta += 360;
    state.perimeterStart.totalRotation += angleDelta;
    state.perimeterStart.lastAngle = state.robot.angle;
    state.perimeterStart.moves++;
    
    // Completion: Near start + made full rotation (~360°) + minimum moves
    const distToStart = Math.hypot(state.robot.x - state.perimeterStart.x, 
                                   state.robot.y - state.perimeterStart.y);
    const madeFullCircle = Math.abs(state.perimeterStart.totalRotation) > 300;
    
    if (state.perimeterStart.hugging && state.perimeterStart.moves > 40 && 
        distToStart < 1.2 && madeFullCircle) {
      console.log('✓ Perimeter complete. Starting snake...');
      state.phase = 'SNAKE';
      // Align to East/West for snake
      if (state.robot.angle > 45 && state.robot.angle <= 135) return 'left';  // Face North->West
      if (state.robot.angle > 135 && state.robot.angle <= 225) return 'right'; // Face West->North then left?
      if (state.robot.angle > 225 && state.robot.angle <= 315) return 'right'; // Face South->West
      // Already roughly East/West
      state.snakeDirection = (state.robot.angle < 90 || state.robot.angle > 270) ? 1 : -1;
      return 'forward';
    }

    // Step 1: Find a wall to hug
    if (!state.perimeterStart.hugging) {
      if (state.distances.Forward > 0.6 && state.distances.Right > 0.6) {
        return 'forward';
      }
      state.perimeterStart.hugging = true;
      return (state.distances.Forward <= 0.6) ? 'right' : 'forward';
    }

    // Step 2: Follow wall on right
    const wallRight = state.distances.Right <= 0.6;
    const wallFront = state.distances.Forward <= 0.6;
    
    // Outside corner: had wall, now gone -> turn right to follow new wall
    if (state.hadWallOnRight && !wallRight) {
      state.hadWallOnRight = wallRight;
      return 'right';
    }
    
    // Standard follow: wall on right, clear ahead -> forward
    if (wallRight && !wallFront) {
      state.hadWallOnRight = wallRight;
      return 'forward';
    }
    
    // Inside corner: wall ahead (and wall on right) -> turn left
    if (wallFront && wallRight) {
      state.hadWallOnRight = wallRight;
      return 'left';
    }
    
    // Lost wall (drifted away) -> turn right to find it again
    if (!wallRight) {
      state.hadWallOnRight = wallRight;
      return 'right';
    }
    
    // Dead end
    state.hadWallOnRight = wallRight;
    return 'left';
  }
    // PHASE 2: SNAKE (inside getAlgorithmicCommand)
  if (state.phase === 'SNAKE') {
    const hitWall = state.distances.Forward < 0.6;
    
    if (hitWall) {
      // Determine which way to turn based on current heading
      const goingEast = state.robot.angle === 0;
      const goingWest = state.robot.angle === 180;
      
      if (goingEast || goingWest) {
        // Try to move down (or up) to next row
        // Check both down (90) and up (270) to find an open path
        const canGoDown = !checkCollision(
          state.robot.x + Math.cos(toRad(90)) * MOVE_STEP,
          state.robot.y + Math.sin(toRad(90)) * MOVE_STEP
        );
        const canGoUp = !checkCollision(
          state.robot.x + Math.cos(toRad(270)) * MOVE_STEP,
          state.robot.y + Math.sin(toRad(270)) * MOVE_STEP
        );
        
        if (canGoDown || canGoUp) {
          state.snakeDirection *= -1;
          // Turn toward the open direction
          if (goingEast) return canGoDown ? 'right' : 'left';
          else return canGoDown ? 'left' : 'right';
        } else {
          // Completely stuck - coverage complete or blocked
          state.completed = true;
          return 'pass';
        }
      }
    }
    
    // Continue in current snake direction...
    const targetAngle = state.snakeDirection === 1 ? 0 : 180;
    if (state.robot.angle === targetAngle) return 'forward';
    
    const diff = (targetAngle - state.robot.angle + 360) % 360;
    if (diff === 90) return 'right';
    if (diff === 270) return 'left';
    return 'right';
  }

  return 'pass';
}

// --- AI Logic ---
async function getAICommand() {
  await new Promise(r => setTimeout(r, 800));
  
  const available = ['right', 'left', 'pass'];
  if (state.distances.Forward > 0.6) available.push('forward');
  
  const prompt = `You are a robot vacuum. Telemetry:
- Position: (${state.robot.x.toFixed(1)}, ${state.robot.y.toFixed(1)})
- Facing: ${state.robot.angle}°
- Forward: ${state.distances.Forward.toFixed(2)}m${state.distances.Forward <= 0.6 ? ' [BLOCKED]' : ''}
- Left: ${state.distances.Left.toFixed(2)}m
- Right: ${state.distances.Right.toFixed(2)}m
- Coverage: ${getCoverageStats().percent}%
- Phase: ${state.phase}

Available commands: ${available.join(', ')}
Strategy: Circle room perimeter first, then fill interior with back-forth pattern.
Output exactly one command.`;

  try {
    const response = romai.ask(prompt, aiModel, "Robot controller. Output one word.", false);
    const clean = response.toLowerCase().trim();
    
    for (let cmd of available) {
      if (clean.includes(cmd)) return cmd;
    }
    return available[0];
  } catch (e) {
    return 'pass';
  }
}

// --- Rendering ---
function draw() {
  rl.BeginDrawing();
  rl.ClearBackground(rl.Color(240, 240, 240, 255));
  
  for (let y = 0; y < GRID_CELLS; y++) {
    for (let x = 0; x < GRID_CELLS; x++) {
      if (state.cleaned[y][x]) {
        rl.DrawRectangle(
          x * GRID_RES * SCALE, 
          y * GRID_RES * SCALE, 
          GRID_RES * SCALE - 1, 
          GRID_RES * SCALE - 1, 
          rl.Color(100, 255, 100, 80)
        );
      }
    }
  }
  
  for (let i = 0; i <= ROOM_SIZE; i++) {
    const pos = i * SCALE;
    rl.DrawLine(pos, 0, pos, WIN_SIZE, rl.Color(200, 200, 200, 255));
    rl.DrawLine(0, pos, WIN_SIZE, pos, rl.Color(200, 200, 200, 255));
  }
  
  const cx = (state.chair.x - state.chair.width/2) * SCALE;
  const cy = (state.chair.y - state.chair.height/2) * SCALE;
  rl.DrawRectangle(cx, cy, state.chair.width*SCALE, state.chair.height*SCALE, rl.BROWN);
  rl.DrawRectangleLines(cx, cy, state.chair.width*SCALE, state.chair.height*SCALE, rl.DARKBROWN);
  
  const rx = state.robot.x * SCALE;
  const ry = state.robot.y * SCALE;
  const r = (ROBOT_SIZE/2) * SCALE;
  
  rl.DrawCircle(rx, ry, r, isBot ? (aiModel ? rl.PURPLE : rl.Color(0, 150, 255, 255)) : rl.BLUE);
  rl.DrawCircleLines(rx, ry, r, rl.DARKBLUE);
  
  const vec = getVector(state.robot.angle);
  rl.DrawLine(rx, ry, rx + vec.x*r*2, ry + vec.y*r*2, rl.RED);
  
  ['Forward', 'Left', 'Right'].forEach(name => {
    const ang = name === 'Forward' ? state.robot.angle : 
                name === 'Left' ? state.robot.angle - 90 : state.robot.angle + 90;
    const v = getVector(ang);
    const dist = state.distances[name];
    const blocked = name === 'Forward' && dist < 0.6;
    const color = blocked ? rl.RED : (name === 'Forward' ? rl.GREEN : rl.ORANGE);
    const endX = rx + v.x * dist * SCALE;
    const endY = ry + v.y * dist * SCALE;
    
    rl.DrawLine(rx, ry, endX, endY, color);
    rl.DrawCircle(endX, endY, 3, rl.RED);
  });
  
  const stats = getCoverageStats();
  const modeText = isBot ? (aiModel ? `AI:${aiModel}` : `${state.phase}`) : 'MANUAL';
  rl.DrawText(`Mode: ${modeText}`, 10, 10, 20, isBot ? rl.Color(0, 200, 255, 255) : rl.BLACK);
  rl.DrawText(`Moves: ${state.moveCount} | Cleaned: ${stats.percent}%`, 10, 35, 20, rl.DARKGRAY);
  
  if (state.completed) {
    rl.DrawText("100% COVERAGE!", WIN_SIZE/2 - 100, WIN_SIZE/2 - 20, 30, rl.GREEN);
  }
  
  rl.EndDrawing();
}

// --- Terminal UI ---
function printTelemetry() {
  const d = state.distances;
  const stats = getCoverageStats();
  const mode = isBot ? (aiModel ? `AI (${aiModel})` : `Algorithm [${state.phase}]`) : 'Human';
  
  const output = `
╔══════════════════════════════════════════╗
║        ROBOT VACUUM TELEMETRY            ║
║        Mode: ${mode.padEnd(26)} ║
╠══════════════════════════════════════════╣
║  Distance to objects:                    ║
║    Forward - ${d.Forward.toFixed(2).padStart(5)}m ${d.Forward < 0.6 ? '[BLOCKED]' : '         '}       ║
║    Left    - ${d.Left.toFixed(2).padStart(5)}m                  ║
║    Right   - ${d.Right.toFixed(2).padStart(5)}m                  ║
╠══════════════════════════════════════════╣
║  Coverage: ${stats.percent}% (${stats.cleaned}/${stats.total} cells)        ║
║  Position: ${state.robot.x.toFixed(1)},${state.robot.y.toFixed(1)}m  Angle: ${state.robot.angle.toString().padStart(3)}°      ║
╠══════════════════════════════════════════╣
${isBot ? 
`║  🤖 Phase: ${state.phase.padEnd(24)} ║` : 
'║  Commands: right | left | forward | pass ║'}
${state.completed ? '║  ✅ COMPLETE                             ║' : ''}
╚══════════════════════════════════════════╝
`;
  console.log(output);
}

// --- Input Handling ---
const rlInterface = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function waitForHumanCommand() {
  return new Promise((resolve) => {
    const interval = setInterval(() => {
      if (rl.WindowShouldClose()) {
        clearInterval(interval);
        resolve('quit');
      }
      draw();
    }, 16);
    
    rlInterface.question('> ', (answer) => {
      clearInterval(interval);
      resolve(answer.trim().toLowerCase());
    });
  });
}

async function getBotCommand() {
  if (aiModel !== null) {
    return await getAICommand();
  } else {
    await new Promise(r => setTimeout(r, 200));
    return getAlgorithmicCommand();
  }
}

// --- Main Loop ---
async function main() {
  rl.InitWindow(WIN_SIZE, WIN_SIZE, `Robot Vacuum - ${isBot ? 'Auto' : 'Manual'}`);
  rl.SetTargetFPS(60);
  
  markCleaned();
  updateSensors();
  draw();
  printTelemetry();
  
  while (!rl.WindowShouldClose()) {
    let cmd;
    
    if (isBot) {
      cmd = await getBotCommand();
      if (cmd !== 'pass' || !state.completed) {
        console.log(`> ${cmd}`);
      }
    } else {
      cmd = await waitForHumanCommand();
      if (cmd === 'quit' || cmd === 'exit') break;
    }
    
    if (['right', 'left', 'forward', 'pass'].includes(cmd)) {
      if (!isBot && cmd === 'forward' && state.distances.Forward < 0.6) {
        console.log('*** Forward blocked! Turn first. ***');
        continue;
      }
      
      executeCommand(cmd);
      updateSensors();
      draw();
      printTelemetry();
    } else if (!isBot) {
      console.log(`? Unknown: "${cmd}"`);
    }
    
    if (!state.completed && isBot && !aiModel) {
      const stats = getCoverageStats();
      if (stats.percent >= 99.5) {
        state.completed = true;
        console.log('\n╔════════════════════════════════╗');
        console.log('║   🎉 100% COVERAGE ACHIEVED!   ║');
        console.log(`║   Total moves: ${state.moveCount.toString().padStart(16)} ║`);
        console.log('╚════════════════════════════════╝\n');
      }
    }
  }
  
  rl.CloseWindow();
  rlInterface.close();
  process.exit(0);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});

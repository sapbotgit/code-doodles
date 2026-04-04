-- Rotating Cube Demo - Fixed Cursor & Clear for Windows Terminal

local isCC = false
local mon

-- ComputerCraft detection
if peripheral and peripheral.wrap then
    mon = peripheral.wrap("right")
    if mon then
        isCC = true
        print("Using Advanced Monitor on the right")
    end
end

if not mon and term and term.clear then
    mon = term
    isCC = true
    print("Using ComputerCraft terminal")
end

-- Plain Lua with ANSI support (best for Windows Terminal / PowerShell)
if not mon then
    local useANSI = true

    local function clearScreen()
        if useANSI then
            io.write("\027[2J\027[H")  -- Clear screen + move cursor to top-left
        else
            for i = 1, 45 do print() end
        end
    end

    mon = {
        clear = clearScreen,
        setCursorPos = function() end,  -- not needed anymore
        write = function(text) io.write(text) end,
        setTextColor = function() end,
    }

    print("Running in plain Lua with ANSI clear (Windows Terminal recommended)")
    print("If the screen doesn't clear properly, change 'useANSI = true' to false in the code.")
end

-- Auto text scale for real monitors
local isMonitor = isCC and mon and mon.setTextScale

local function autoSetScale(display)
    if not isMonitor then return end
    local w, h = display.getSize()
    if w >= 80 and h >= 40 then
        display.setTextScale(3)
    elseif w >= 50 and h >= 25 then
        display.setTextScale(2)
    else
        display.setTextScale(1)
    end
end

autoSetScale(mon)

local function getSize()
    if mon.getSize then
        return mon.getSize()
    else
        return 78, 28
    end
end

local width, height = getSize()

-- Cube
local vertices = {
    {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
    {-1,-1,1},  {1,-1,1},  {1,1,1},  {-1,1,1}
}

local edges = {
    {1,2},{2,3},{3,4},{4,1},
    {5,6},{6,7},{7,8},{8,5},
    {1,5},{2,6},{3,7},{4,8}
}

local function rotateY(p, a)
    local c, s = math.cos(a), math.sin(a)
    return {p[1]*c - p[3]*s, p[2], p[1]*s + p[3]*c}
end

local function rotateX(p, a)
    local c, s = math.cos(a), math.sin(a)
    return {p[1], p[2]*c - p[3]*s, p[2]*s + p[3]*c}
end

local function project(p)
    local scale = math.min(width, height) / 4.6
    local x = math.floor(width / 2 + p[1] * scale)
    local y = math.floor(height / 2 - p[2] * scale)
    return x, y
end

-- Build full frame as one string (prevents cursor issues)
local function drawFrame(angle)
    local frame = {}  -- table of lines

    -- Fill with spaces
    for y = 1, height do
        frame[y] = string.rep(" ", width)
    end

    -- Rotate
    local rotated = {}
    for i, v in ipairs(vertices) do
        local p = rotateY(v, angle)
        p = rotateX(p, angle * 0.72)
        rotated[i] = p
    end

    -- Draw edges sparsely
    for _, e in ipairs(edges) do
        local p1 = rotated[e[1]]
        local p2 = rotated[e[2]]
        local x1, y1 = project(p1)
        local x2, y2 = project(p2)

        local dx = x2 - x1
        local dy = y2 - y1
        local dist = math.max(math.abs(dx), math.abs(dy))
        if dist == 0 then goto continue end

        local step = 2.0  -- controls spacing (higher = more space)

        for i = 0, dist, step do
            local t = i / dist
            local x = math.floor(x1 + dx * t + 0.5)
            local y = math.floor(y1 + dy * t + 0.5)

            if x >= 3 and x <= width-2 and y >= 3 and y <= height-3 then
                local line = frame[y]
                frame[y] = line:sub(1, x-1) .. "#" .. line:sub(x+1)
            end
        end
        ::continue::
    end

    -- Title
    local title = "=== Rotating Cube Demo ==="
    local tx = math.floor((width - #title) / 2)
    if tx < 1 then tx = 1 end
    frame[2] = frame[2]:sub(1, tx-1) .. title .. frame[2]:sub(tx + #title)

    -- Center point
    local cx = math.floor(width/2)
    local cy = math.floor(height/2)
    if cx >= 1 and cx <= width and cy >= 1 and cy <= height then
        local line = frame[cy]
        frame[cy] = line:sub(1, cx-1) .. "+" .. line:sub(cx+1)
    end

    -- Footer
    local footer = "Press Ctrl+C to stop"
    local fx = 3
    frame[height] = frame[height]:sub(1, fx-1) .. footer .. frame[height]:sub(fx + #footer)

    -- Convert to single string with newlines
    return table.concat(frame, "\n")
end

-- Main loop
local angle = 0
local speed = 0.07

print("Starting Rotating Cube... Press Ctrl+C to stop\n")

while true do
    mon.clear()

    local frameText = drawFrame(angle)
    mon.write(frameText)

    angle = angle + speed

    if isCC and sleep then
        sleep(0.07)
    else
        local t = os.clock()
        while os.clock() - t < 0.07 do end
    end
end

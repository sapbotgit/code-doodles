#!/bin/bash

# Cookie Clicker Automation Script
# Periodically clicks cookie and buys items

# Coordinates from README_COOKIECLICKER.md
COOKIE_SECTION_X=61
COOKIE_SECTION_Y=904
COOKIE_X=270
COOKIE_Y=480
STORE_X=142
STORE_Y=900
BUY_CLICKER_X=416
BUY_CLICKER_Y=171
BUY_GRANDMA_X=422
BUY_GRANDMA_Y=252
BUY_FARM_X=418
BUY_FARM_Y=344
BUY_MINE_X=416
BUY_MINE_Y=419
BUY_FACTORY_X=424
BUY_FACTORY_Y=498

# Function to click on coordinates
click() {
    local x=$1
    local y=$2
    adb shell input tap "$x" "$y"
}

# Function to buy items if we can afford them
buy_items() {
    # Click on store section
    click "$STORE_X" "$STORE_Y"
    sleep 1

    # Try to buy clicker
    click "$BUY_CLICKER_X" "$BUY_CLICKER_Y"
    sleep 1

    # Try to buy grandma
    click "$BUY_GRANDMA_X" "$BUY_GRANDMA_Y"
    sleep 1

    # Try to buy farm
    click "$BUY_FARM_X" "$BUY_FARM_Y"
    sleep 1

    # Try to buy mine
    click "$BUY_MINE_X" "$BUY_MINE_Y"
    sleep 1

    # Try to buy factory
    click "$BUY_FACTORY_X" "$BUY_FACTORY_Y"
    sleep 1
}

# Main automation loop
main() {
    echo "Starting Cookie Clicker automation..."
    echo "Will periodically click cookie 10 times and buy items"

    iteration=0
    while true; do
        iteration=$((iteration + 1))
        echo "Iteration: $iteration"

        # First navigate to cookie section
        click "$COOKIE_SECTION_X" "$COOKIE_SECTION_Y"
        sleep 1

        # Click on cookie 10 times
        for i in {1..10}; do
            click "$COOKIE_X" "$COOKIE_Y"
            sleep 0.3
        done

        # Try to buy items
        buy_items

        # Longer delay between cycles
        sleep 3
    done
}

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo "Error: adb not found. Please install Android SDK platform tools."
    exit 1
fi

# Run the automation
main

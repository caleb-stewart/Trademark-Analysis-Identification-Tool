#!/bin/bash

# === FUNCTION: Check if a command exists ===
check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo "❌ '$1' is not installed."
    exit 1
  fi
}

# === FUNCTION: Print section header ===
print_section() {
  echo ""
  echo "==============================="
  echo "📂 $1"
  echo "==============================="
}

# === FUNCTION: Cleanup on exit or interrupt ===
cleanup() {
  echo ""
  echo "Shutting down..."
  kill $FRONT_PID 2>/dev/null && echo "🛑 Frontend stopped."
  kill $BACK_PID 2>/dev/null && echo "🛑 Backend stopped."
  deactivate 2>/dev/null
  echo "✅ All processes terminated. Goodbye!"
}

# === SET TRAP ===
trap cleanup EXIT INT TERM

# === CHECK PREREQUISITES ===
print_section "Checking prerequisites"
check_command python3
check_command pip
check_command npm

if ! python3 -m venv --help > /dev/null 2>&1; then
  echo "❌ Python 'venv' module is not available."
  exit 1
fi

echo "✅ All prerequisites met."

# === FRONTEND SETUP ===
print_section "Setting up frontend"
cd TRAIT-Front

if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
else
  echo "✅ Frontend dependencies already installed."
fi

echo "Starting frontend..."
npm run start &
FRONT_PID=$!
cd - > /dev/null

# === BACKEND SETUP ===
print_section "Setting up backend"
cd TRAIT-Back

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies if not already installed
if [ ! -d "venv/lib" ] || [ -z "$(ls -A venv/lib/*/site-packages)" ]; then
  echo "Installing backend Python dependencies..."
  pip install -r requirements.txt
else
  echo "✅ Backend dependencies already installed."
fi

echo "Starting backend..."
python3 run.py &
BACK_PID=$!
cd - > /dev/null

# === USER PROMPT ===
echo ""
echo "Press [Enter] to stop both servers..."
read
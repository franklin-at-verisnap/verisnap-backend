#!/bin/bash

# Verisnap Backend Virtual Environment Setup Script

echo "Setting up Verisnap Backend virtual environment..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Verisnap Backend Configuration
API_KEY=your_google_maps_api_key_here
APPLE_KID=your_apple_kid_here
APPLE_TEAM_ID=your_apple_team_id_here
PRIVATE_KEY_PATH=path_to_your_apple_private_key.p8

# Server Configuration
HOST=0.0.0.0
PORT=9000
DEBUG=false

# Static Files
ENABLE_STATIC_SERVICE=true

# ML Model Configuration
DEVICE=auto
EOF
    echo "Please edit .env file with your actual configuration values."
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p storage
mkdir -p db
mkdir -p models
mkdir -p static

echo "Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application, run:"
echo "  python -m app.main"
echo ""
echo "Or with uvicorn directly:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload"

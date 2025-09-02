# Verisnap Backend

A sophisticated image verification and truth scoring service built with FastAPI. This service analyzes uploaded images using machine learning models to detect authenticity, verify location data, and calculate trust scores.

## Features

- **Image Verification**: Advanced ML-based analysis using CLIP and MiDaS models
- **Truth Scoring**: Multi-factor scoring system including:
  - Timestamp verification
  - Location and altitude validation
  - Device authentication via Apple DeviceCheck
  - Magnetometer and barometer analysis
  - Indoor/outdoor scene detection
  - Photo-of-photo detection
- **Authentication**: Firebase ID token verification
- **Static File Serving**: Optional thumbnail browser interface
- **RESTful API**: FastAPI-based with automatic OpenAPI documentation
- **Production Ready**: Configurable for development and production environments

## Architecture

The application has been restructured from the original Flask implementation into a modular FastAPI architecture:

```
verisnap-backend/
├── app/
│   ├── models/          # Pydantic data models
│   ├── utils/           # Service utilities
│   │   ├── image_processing.py
│   │   ├── geolocation.py
│   │   ├── scoring.py
│   │   ├── database.py
│   │   └── auth.py
│   └── main.py          # FastAPI application
├── config/              # Configuration management
├── static/              # Static files (HTML, CSS, JS)
├── storage/             # User image storage
├── db/                  # SQLite database
├── models/              # ML model files
└── requirements.txt     # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- Google Maps API key (for geolocation services)
- Apple Developer account (for DeviceCheck API)
- Firebase project (for authentication)

## Installation

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd verisnap-backend

# Run the setup script
./setup_venv.sh
```

### 2. Manual Setup (Alternative)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit the `.env` file with your configuration:

```env
# Google Maps API
API_KEY=your_google_maps_api_key_here

# Apple DeviceCheck
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
```

### 4. Required Files

Ensure you have the following files in the project root:
- `verisnap-poc-firebase-adminsdk-fbsvc-d8f2304cdb.json` (Firebase credentials)
- `models/p2p_classifier.joblib` (Photo-of-photo classifier model)
- Apple private key file (`.p8` format)

## Running the Application

### Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run with static service enabled
python run_server.py --static --reload --debug

# Or run directly with uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```

### Production Mode

```bash
# Run without static service (recommended for production)
python run_server.py --no-static --workers 4

# Or with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 9000 --workers 4
```

### Command Line Options

```bash
python run_server.py --help
```

Available options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 9000)
- `--debug`: Enable debug mode
- `--static`: Enable static file serving
- `--no-static`: Disable static file serving
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes

## API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: http://localhost:9000/docs
- **ReDoc Documentation**: http://localhost:9000/redoc
- **OpenAPI Schema**: http://localhost:9000/openapi.json

## API Endpoints

### Core Endpoints

- `GET /ping` - Health check
- `POST /upload` - Upload and verify image (requires authentication)
- `POST /check` - Check if image matches existing verified images (requires authentication)

### Image Serving

- `GET /thumbnail/{userid}/{image_id}.jpeg` - Serve thumbnail
- `GET /image/{userid}/{image_id}.jpeg` - Serve full image
- `GET /thumbnails` - List thumbnails with pagination

### Data Access

- `GET /capture/{userid}/{image_id}` - Get capture metadata
- `GET /me/storage` - Get storage usage report (requires authentication)

### Static Files (if enabled)

- `GET /static/thumbnail_browser.html` - Thumbnail browser interface

## Authentication

The API uses Firebase ID tokens for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <firebase_id_token>
```

## Static Service Configuration

The static service can be enabled or disabled based on your deployment needs:

### Development
```bash
# Enable static service for local development
python run_server.py --static
```

### Production
```bash
# Disable static service for production deployment
python run_server.py --no-static
```

You can also configure this via environment variables:
```env
ENABLE_STATIC_SERVICE=false
```

## Database

The application uses SQLite for data storage. The database is automatically created and initialized on first run.

### Database Schema

The `captures` table stores:
- Image metadata and signatures
- Verification scores
- Location and device data
- Timestamps and user information

## Machine Learning Models

The service uses several ML models:

1. **CLIP (ViT-B/32)**: For scene classification and content analysis
2. **MiDaS (DPT_Large)**: For depth estimation and photo-of-photo detection
3. **Custom P2P Classifier**: Trained model for detecting photos of photos

Models are loaded automatically on startup and cached in memory.

## Scoring System

The truth scoring system evaluates multiple factors:

- **Timestamp Score**: Verifies image capture time vs server time
- **OS Score**: Checks operating system version security
- **Location Score**: Validates GPS coordinates and altitude
- **Device Score**: Apple DeviceCheck verification
- **Magnetometer Score**: Compass and magnetic field validation
- **Scene Analysis**: Indoor/outdoor detection and day/night verification
- **Anti-cheat**: Photo-of-photo detection and manipulation detection

## Development

### Code Structure

The codebase is organized into logical modules:

- **Models**: Pydantic models for request/response validation
- **Services**: Business logic and utility functions
- **Configuration**: Environment-based settings management
- **Main**: FastAPI application and route definitions

### Adding New Features

1. Create new models in `app/models/`
2. Add service logic in `app/utils/`
3. Define routes in `app/main.py`
4. Update configuration as needed

### Testing

```bash
# Run tests (when implemented)
pytest

# Run with coverage
pytest --cov=app
```

### Tooling

- Pre-commit hooks are configured for basic hygiene (trailing whitespace, end-of-file fixer, etc.). To enable locally:
  - Install pre-commit: `pip install pre-commit`
  - Install hooks: `pre-commit install`
  - Run on demand: `pre-commit run --all-files` or `make pre-commit`

- Makefile shortcuts are available:
  - `make install` – install dependencies and set up pre-commit hooks (if available)
  - `make dev` – run the server in development mode (reload + debug + static)
  - `make run` – run the server in production-like mode

## Deployment

### Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 9000

CMD ["python", "run_server.py", "--no-static", "--workers", "4"]
```

### Environment Variables

Set these environment variables in production:

```env
API_KEY=your_production_api_key
APPLE_KID=your_production_apple_kid
APPLE_TEAM_ID=your_production_team_id
PRIVATE_KEY_PATH=/path/to/production/key.p8
ENABLE_STATIC_SERVICE=false
DEBUG=false
HOST=0.0.0.0
PORT=9000
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model files are in the correct location
2. **Authentication Failures**: Verify Firebase credentials and configuration
3. **API Key Issues**: Check Google Maps API key and quotas
4. **Device Check Failures**: Verify Apple Developer configuration

### Logs

Check the application logs for detailed error information:

```bash
# View logs in real-time
tail -f logs/verisnap.log
```

### Performance

For production deployment:
- Use multiple workers (`--workers 4`)
- Disable static service (`--no-static`)
- Set `DEBUG=false`
- Use a reverse proxy (nginx) for static files

## Migration from Flask

This FastAPI version provides several improvements over the original Flask implementation:

- **Better Performance**: Async support and higher throughput
- **Type Safety**: Pydantic models and type hints
- **Auto Documentation**: OpenAPI/Swagger integration
- **Modern Architecture**: Modular design and dependency injection
- **Production Ready**: Better configuration management and deployment options

## License

[Add your license information here]

## Support

For support and questions, please [add contact information].
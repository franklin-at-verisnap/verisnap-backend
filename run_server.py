#!/usr/bin/env python3
"""
Verisnap Backend Server Startup Script
"""
import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings


def main():
    parser = argparse.ArgumentParser(description="Verisnap Backend Server")
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", default=settings.debug, help="Enable debug mode")
    parser.add_argument("--static", action="store_true", default=settings.enable_static_service, 
                       help="Enable static file serving")
    parser.add_argument("--no-static", action="store_true", help="Disable static file serving")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Override static service setting
    if args.no_static:
        settings.enable_static_service = False
    elif args.static:
        settings.enable_static_service = True
    
    # Print configuration
    print("=" * 50)
    print("Verisnap Backend Server")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Static Service: {settings.enable_static_service}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print("=" * 50)
    
    # Ensure required directories exist
    os.makedirs("storage", exist_ok=True)
    os.makedirs("db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Start server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="debug" if args.debug else "info",
        access_log=True
    )


if __name__ == "__main__":
    main()

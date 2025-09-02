"""
Production configuration settings
"""
from .settings import Settings


class ProductionSettings(Settings):
    """Production-specific settings"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    
    # Static Files - Disabled in production
    enable_static_service: bool = False
    
    # Security
    cors_origins: list = []  # Configure specific origins in production
    
    # Performance
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/verisnap.log"
    
    class Config:
        env_file = ".env.production"
        case_sensitive = False


# Production settings instance
production_settings = ProductionSettings()

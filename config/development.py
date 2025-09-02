"""
Development configuration settings
"""
from .settings import Settings


class DevelopmentSettings(Settings):
    """Development-specific settings"""
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 9000
    debug: bool = True
    
    # Static Files - Enabled in development
    enable_static_service: bool = True
    
    # Security - More permissive for development
    cors_origins: list = ["*"]
    
    # Performance
    max_upload_size: int = 50 * 1024 * 1024  # 50MB for development
    request_timeout: int = 60
    
    # Logging
    log_level: str = "DEBUG"
    log_file: str = "logs/verisnap_dev.log"
    
    class Config:
        env_file = ".env.development"
        case_sensitive = False


# Development settings instance
development_settings = DevelopmentSettings()

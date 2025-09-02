"""
Configuration settings for Verisnap Backend
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Verisnap Backend API"
    api_version: str = "1.0.0"
    api_description: str = "Image verification and truth scoring service"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    
    # Static Files Configuration
    enable_static_service: bool = True

    # Public Base URL (used for building absolute URLs in responses)
    public_base_url: str = "http://localhost:9000"
    
    # External API Keys
    api_key: Optional[str] = None  # Google Maps API key
    apple_kid: Optional[str] = None
    apple_team_id: Optional[str] = None
    private_key_path: Optional[str] = None
    
    # Firebase Configuration
    firebase_credentials_path: str = "verisnap-poc-firebase-adminsdk-fbsvc-d8f2304cdb.json"
    
    # Database Configuration
    database_path: str = "db/truths.db"
    
    # Storage Configuration
    storage_base_path: str = "storage"
    
    # Model Configuration
    model_path: str = "models/p2p_classifier.joblib"
    
    # ML Model Configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

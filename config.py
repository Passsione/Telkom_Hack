"""
Configuration file for the chatbot simulator
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Base configuration class"""
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG: bool = True
    
    # Database configuration (if needed later)
    SQLALCHEMY_DATABASE_URI: str = os.environ.get('DATABASE_URL', 'sqlite:///chatbot.db')
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    
    # API Configuration
    API_TIMEOUT: int = 30
    MAX_TOKENS: int = 1000
    DEFAULT_TEMPERATURE: float = 0.7

@dataclass 
class APIConfig:
    """Configuration for different AI APIs"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.environ.get('OPENAI_API_KEY', '')
    OPENAI_MODEL: str = 'gpt-3.5-turbo'
    
    # Anthropic Claude Configuration
    ANTHROPIC_API_KEY: str = os.environ.get('ANTHROPIC_API_KEY', '')
    ANTHROPIC_MODEL: str = 'claude-3-sonnet-20240229'
    
    # Google Gemini Configuration
    GOOGLE_API_KEY: str = os.environ.get('GOOGLE_API_KEY', '')
    GEMINI_MODEL: str = 'gemini-pro'
    
    # Custom API Configuration (for future APIs)
    CUSTOM_API_BASE_URL: str = os.environ.get('CUSTOM_API_BASE_URL', '')
    CUSTOM_API_KEY: str = os.environ.get('CUSTOM_API_KEY', '')

# Initialize configurations
config = Config()
api_config = APIConfig()

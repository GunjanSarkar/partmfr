import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    
    # Databricks connection settings
    databricks_server_hostname: str = os.environ.get("DATABRICKS_SERVER_HOSTNAME", "")
    databricks_http_path: str = os.environ.get("DATABRICKS_HTTP_PATH", "")
    databricks_access_token: str = os.environ.get("DATABRICKS_ACCESS_TOKEN", "")
    databricks_table_name: str = os.environ.get("DATABRICKS_TABLE_NAME", "devteam_poc.default.master_index_with_class")
    
    # For backward compatibility with existing environment variables
    def __init__(self, **data):
        # Map old environment variable names to new attribute names
        if 'server_hostname' in os.environ:
            data['databricks_server_hostname'] = os.environ.get('server_hostname')
        if 'http_path' in os.environ:
            data['databricks_http_path'] = os.environ.get('http_path')
        if 'access_token' in os.environ:
            data['databricks_access_token'] = os.environ.get('access_token')
        
        super().__init__(**data)
    
    # Database configuration - always use Databricks
    use_databricks: bool = True  # Always use Databricks, SQLite support removed
    
    # Connection pooling settings
    db_pool_max_connections: int = 10
    db_pool_timeout: int = 30
    
    # API settings
    openai_model_primary: str = "o4-mini"
    openai_model_fallback1: str = "gpt-4-1106-preview"
    openai_model_fallback2: str = "gpt-4"
    temperature: float = 1.0
    max_tokens_primary: int = 128000    # o4-mini max context
    max_tokens_fallback1: int = 128000  # gpt-4-turbo max context
    max_tokens_fallback2: int = 8000    # gpt-4 base max context
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in environment variables

settings = Settings()

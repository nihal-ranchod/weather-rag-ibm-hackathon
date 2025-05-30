# Configuration for Weather RAG System
import os

# IBM watsonx.ai Configuration
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY', 'your_api_key_here')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID', 'your_project_id_here') 
WATSONX_ENDPOINT = os.getenv('WATSONX_ENDPOINT', 'https://us-south.ml.cloud.ibm.com')

# Flask Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Weather API Configuration
OPEN_METEO_BASE_URL = 'https://api.open-meteo.com/v1'

# Default locations for quick access
DEFAULT_LOCATIONS = [
    {"name": "Miami, FL", "lat": 25.7617, "lon": -80.1918, "description": "Hurricane-prone coastal city"},
    {"name": "Moore, OK", "lat": 35.3395, "lon": -97.4864, "description": "Tornado Alley location"},
    {"name": "Phoenix, AZ", "lat": 33.4484, "lon": -112.0740, "description": "Extreme heat events"},
    {"name": "Buffalo, NY", "lat": 42.8864, "lon": -78.8784, "description": "Lake effect snow and blizzards"},
    {"name": "New Orleans, LA", "lat": 29.9511, "lon": -90.0715, "description": "Hurricane and flood risk"},
]

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/weather_rag.log'

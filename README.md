# Extreme Weather RAG System

**AI-Powered Climate Emergency Response Platform**  
*Built for IBM watsonx.ai Hackathon - Climate Challenge*

[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-blue?style=for-the-badge&logo=ibm)](https://www.ibm.com/watsonx)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)

---

## Project Overview

The **Extreme Weather RAG System** is an intelligent early warning platform that combines real-time weather monitoring, historical data analysis, and AI-powered insights to predict and alert communities about dangerous weather events. Using IBM watsonx.ai's Granite models and 90+ TB of historical weather data, our system provides life-saving predictions and actionable guidance for extreme weather preparedness.

### Hackathon Theme: Climate Challenge
Our solution addresses the urgent need for AI-driven climate adaptation tools that can save lives and protect communities from increasingly severe weather events.

---

## Team Members - CodeX

- **Nihal Ranchod** NRanchod@datacentrix.co.za
- **Eza Ngam** ENgam@datacentrix.co.za
- **Manisha Nankoo** MNankoo@datacentrix.co.za
- **Zakaria Motala** ZMotala@datacentrix.co.za

---

## Key Features

### AI-Powered Analysis
- **IBM watsonx.ai Integration**: Granite models for intelligent weather pattern analysis
- **RAG Architecture**: Retrieval-Augmented Generation for context-aware predictions
- **Natural Language Interface**: Chat with the AI about weather conditions worldwide

### Global Coverage
- **Any Location Worldwide**: Smart geocoding with OpenStreetMap integration
- **90+ TB Historical Data**: Weather patterns from 1940-present via Open-Meteo API
- **Real-time Monitoring**: 7-day forecasts with hourly precision

### Extreme Weather Detection
- **Hurricane/Typhoon Tracking**: Advanced cyclonic storm prediction
- **Tornado Risk Assessment**: Atmospheric instability analysis
- **Heat Wave/Cold Wave Alerts**: Temperature extreme identification
- **Flash Flood Warnings**: Precipitation-based risk modeling
- **Severe Storm Tracking**: Thunderstorm and hail prediction

### User Experience
- **Real-time Chat Interface**: WebSocket-powered instant communication
- **Community Alerts**: Plain-language emergency notifications
- **Mobile-Responsive Design**: Works on any device
- **Command System**: Quick analysis with simple commands

---

## System Architecture

```
User Interface → Flask Web App → Weather RAG System → IBM watsonx.ai
                              ↓
                Weather Data Collector → Open-Meteo API (90TB+ Historical Data)
                              ↓
                Extreme Weather Detector → Prediction Engine
```

### Core Components

1. **Weather Data Collection**: Open-Meteo Archive API (1940-present)
2. **Extreme Weather Detection**: Statistical analysis of weather trends
3. **IBM watsonx.ai Integration**: Granite 13B Instruct v2 model
4. **Flask Web Application**: Real-time chat interface

---

## Quick Start

### Prerequisites
- Python 3.8+
- IBM watsonx.ai account with API access

### 1. Clone Repository
```bash
git clone https://github.com/your-team/extreme-weather-rag.git
cd extreme-weather-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file:
```env
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_ENDPOINT=https://us-south.ml.cloud.ibm.com
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Interface
Open your browser to `http://localhost:5000`

---

## Usage Guide

### Commands
- `/analyze [location]` - Comprehensive weather risk analysis
- `/predict [coordinates]` - Location-based prediction
- `/quick [location]` - Fast weather summary
- `/help` - Show all available commands

### Example Queries
```
Hurricane risk for Miami this week?
Tornado warnings for Oklahoma City?
Heat wave predictions for Phoenix?
/predict 25.7617 -80.1918
/analyze Tokyo, Japan
```

### Natural Language Processing
The system understands natural language:
- "What's the weather like in Sydney?"
- "Are there any storm warnings for London?"
- "Show me hurricane predictions for the Gulf Coast"

---

## AI Response Format

The system provides focused weather intelligence in four key sections:

1. **Current Conditions**: Live weather data using RAG
2. **7-Day Forecast Summary**: Concise daily predictions
3. **Extreme Weather Predictions**: Intelligent threat assessment
4. **Weather Intelligence Summary**: General risk analysis for the area

This focused approach ensures users get actionable information without overwhelming detail.

---

## Technical Implementation

### Data Sources
- **Historical Data**: Open-Meteo Archive API (1940-present)
- **Forecast Data**: 7-day predictions with hourly resolution
- **Global Coverage**: Any coordinate worldwide
- **AI Insights**: IBM watsonx.ai Granite models

### Security & Privacy
- **No Personal Data Storage**: Location queries are not logged
- **API Key Security**: Environment variable configuration
- **Open Data Sources**: Publicly available weather information
- **Privacy by Design**: Minimal data collection

---

## Development

### Project Structure
```
extreme-weather-rag/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (create this)
├── src/
│   ├── config.py            # Configuration settings
│   ├── weather_rag_system.py # Core RAG system
│   └── enhanced_ai_response.py # AI response generator
├── static/
│   ├── css/style.css        # Clean styling
│   └── js/chat.js           # Frontend JavaScript
├── templates/
│   └── index.html           # Main HTML template
└── logs/                    # Application logs
```

### Development Setup
1. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Development Mode**:
   ```bash
   export FLASK_DEBUG=1  # On Windows: set FLASK_DEBUG=1
   python app.py
   ```

3. **Testing**:
   ```bash
   python test_system.py
   ```

---

## API Documentation

### Core Endpoints

#### Weather Analysis
```python
POST /api/analyze
{
    "location": "Miami, FL",
    "days_history": 30
}
```

#### System Health
```python
GET /api/health
Response: {
    "status": "healthy",
    "rag_system_ready": true,
    "weather_api_ready": true,
    "timestamp": "2025-05-30T..."
}
```

---

## Impact & Use Cases

### Extreme Weather Event Prediction

Our RAG solution directly addresses the critical need for accurate extreme weather prediction by combining historical analysis with real-time monitoring.

#### Hurricane & Typhoon Prediction
Advanced tracking using historical patterns and current atmospheric conditions.

#### Tornado Risk Assessment  
Early warning system based on atmospheric instability indicators.

#### Heat Wave Early Warning
Temperature extreme detection with health impact assessments.

#### Flash Flood Prediction
Precipitation analysis with terrain and drainage considerations.

---

## License & Attribution

### License
This project is licensed under the MIT License.

### Data Sources
- **Weather Data**: [Open-Meteo API](https://open-meteo.com/) - Open source weather API
- **Geocoding**: [OpenStreetMap Nominatim](https://nominatim.org/) - Free geocoding service
- **AI Models**: [IBM watsonx.ai](https://www.ibm.com/watsonx) - Granite foundation models

### Acknowledgments
- IBM watsonx.ai team for providing AI infrastructure
- Open-Meteo for comprehensive weather data access
- OpenStreetMap community for global geocoding services
- Flask and SocketIO communities for web framework support

---

*Built for the IBM watsonx.ai Hackathon - Climate Challenge*  
*Making communities safer through AI-powered weather intelligence.*
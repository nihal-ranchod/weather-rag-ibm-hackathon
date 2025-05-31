#!/usr/bin/env python3
"""
Weather RAG Chat Interface - Clean Version
Flask web application for extreme weather predictions
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import logging
import os
import sys
from datetime import datetime, timedelta
import traceback
import requests
import re
from typing import Optional, Tuple
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from weather_rag_system import ExtremeWeatherRAGSystem, WeatherDataCollector, ExtremeWeatherDetector
    from config import *
    from enhanced_ai_response import enhance_rag_system_responses, EnhancedWeatherAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are present")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weather_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'weather_rag_secret_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
rag_system = None
weather_collector = WeatherDataCollector()
detector = ExtremeWeatherDetector()

class LocationService:
    """Handles location geocoding and coordinate lookup"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        
    def get_coordinates_and_name(self, location_input: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Get coordinates with fallback strategies"""
        
        cache_key = location_input.lower().strip()
        
        if cache_key in self.cache:
            logger.info(f"Using cached location data for {location_input}")
            return self.cache[cache_key]
        
        logger.info(f"Looking up coordinates for: {location_input}")
        
        # Try known locations first
        known_result = self._try_known_locations(location_input)
        if known_result[0] is not None:
            self.cache[cache_key] = known_result
            return known_result
        
        # Try geocoding services
        for service_name, service_func in [
            ("Nominatim", self._try_nominatim),
            ("Photon", self._try_photon)
        ]:
            try:
                logger.debug(f"Trying {service_name} geocoding service")
                lat, lon, name = service_func(location_input)
                if lat is not None and lon is not None:
                    logger.info(f"Successfully geocoded {location_input} using {service_name}")
                    self.cache[cache_key] = (lat, lon, name)
                    return lat, lon, name
            except Exception as e:
                logger.debug(f"{service_name} geocoding failed: {e}")
                continue
        
        logger.warning(f"Could not geocode location: {location_input}")
        return None, None, None
    
    def _try_known_locations(self, location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try known location database first"""
        
        location_lower = location.lower().strip()
        
        known_locations = {
            # US Cities
            'miami': (25.7617, -80.1918, 'Miami, FL'),
            'miami, fl': (25.7617, -80.1918, 'Miami, FL'),
            'phoenix': (33.4484, -112.0740, 'Phoenix, AZ'),
            'phoenix, az': (33.4484, -112.0740, 'Phoenix, AZ'),
            'oklahoma city': (35.4676, -97.5164, 'Oklahoma City, OK'),
            'oklahoma': (35.4676, -97.5164, 'Oklahoma City, OK'),
            'new york': (40.7128, -74.0060, 'New York, NY'),
            'los angeles': (34.0522, -118.2437, 'Los Angeles, CA'),
            'chicago': (41.8781, -87.6298, 'Chicago, IL'),
            'houston': (29.7604, -95.3698, 'Houston, TX'),
            'new orleans': (29.9511, -90.0715, 'New Orleans, LA'),
            'buffalo': (42.8864, -78.8784, 'Buffalo, NY'),
            
            # International Cities
            'london': (51.5074, -0.1278, 'London, UK'),
            'london, uk': (51.5074, -0.1278, 'London, UK'),
            'tokyo': (35.6762, 139.6503, 'Tokyo, Japan'),
            'tokyo, japan': (35.6762, 139.6503, 'Tokyo, Japan'),
            'sydney': (-33.8688, 151.2093, 'Sydney, Australia'),
            'sydney, australia': (-33.8688, 151.2093, 'Sydney, Australia'),
            'paris': (48.8566, 2.3522, 'Paris, France'),
            'paris, france': (48.8566, 2.3522, 'Paris, France'),
            'reykjavik': (64.1466, -21.9426, 'Reykjavik, Iceland'),
            'reykjavik, iceland': (64.1466, -21.9426, 'Reykjavik, Iceland'),
            
            # Brazil Cities
            'rio': (-22.9068, -43.1729, 'Rio de Janeiro, Brazil'),
            'rio de janeiro': (-22.9068, -43.1729, 'Rio de Janeiro, Brazil'),
            'brasilia': (-15.8267, -47.9218, 'Brasília, Brazil'),
            'brasília': (-15.8267, -47.9218, 'Brasília, Brazil'),
            'sao paulo': (-23.5505, -46.6333, 'São Paulo, Brazil'),
            'são paulo': (-23.5505, -46.6333, 'São Paulo, Brazil'),
            
            # Other major cities
            'berlin': (52.5200, 13.4050, 'Berlin, Germany'),
            'madrid': (40.4168, -3.7038, 'Madrid, Spain'),
            'rome': (41.9028, 12.4964, 'Rome, Italy'),
            'moscow': (55.7558, 37.6176, 'Moscow, Russia'),
            'beijing': (39.9042, 116.4074, 'Beijing, China'),
            'mumbai': (19.0760, 72.8777, 'Mumbai, India'),
            'cairo': (30.0444, 31.2357, 'Cairo, Egypt'),
            'lagos': (6.5244, 3.3792, 'Lagos, Nigeria'),
            
            # Regions
            'gulf coast': (29.0, -90.0, 'Gulf Coast Region'),
        }
        
        return known_locations.get(location_lower, (None, None, None))
    
    def _try_nominatim(self, location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try OpenStreetMap Nominatim"""
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "accept-language": "en"
        }
        headers = {"User-Agent": "WeatherRAGSystem/1.0"}
        
        response = self.session.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                address = result.get('address', {})
                name_parts = []
                
                if address.get('city'):
                    name_parts.append(address['city'])
                elif address.get('town'):
                    name_parts.append(address['town'])
                elif address.get('village'):
                    name_parts.append(address['village'])
                
                if address.get('country_code') == 'us' and address.get('state'):
                    name_parts.append(address['state'])
                elif address.get('country'):
                    name_parts.append(address['country'])
                
                name = ', '.join(name_parts) if name_parts else location.title()
                return lat, lon, name
        
        return None, None, None
    
    def _try_photon(self, location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try Photon geocoding"""
        
        url = "https://photon.komoot.io/api/"
        params = {"q": location, "limit": 1}
        
        response = self.session.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('features'):
                feature = data['features'][0]
                coords = feature['geometry']['coordinates']
                lon, lat = coords[0], coords[1]
                
                props = feature.get('properties', {})
                name_parts = []
                
                if props.get('name'):
                    name_parts.append(props['name'])
                if props.get('state'):
                    name_parts.append(props['state'])
                elif props.get('country'):
                    name_parts.append(props['country'])
                
                name = ', '.join(name_parts) if name_parts else location.title()
                return lat, lon, name
        
        return None, None, None

# Initialize location service
location_service = LocationService()

def parse_location_from_text(text: str) -> Optional[str]:
    """Extract location from natural language"""
    
    text_lower = text.lower()
    
    patterns = [
        r'(?:weather|conditions|forecast|risk|predictions?|warnings?)\s+(?:for|in|at|near)\s+([^?.,!]+)',
        r'(?:hurricane|tornado|storm|heat\s+wave|flood)\s+(?:risk|threat|warning)\s+(?:for|in|at)\s+([^?.,!]+)',
        r'(?:analyze|check|monitor)\s+([^?.,!]+)',
        r'(?:what\'s|how\'s)\s+(?:the\s+)?weather\s+(?:like\s+)?(?:in|at|for)\s+([^?.,!]+)',
        r'predictions?\s+(?:for|in)\s+([^?.,!]+)',
        r'(?:show\s+me|tell\s+me\s+about)\s+.*?(?:for|in|at)\s+([^?.,!]+)',
        r'weather\s+in\s+([^?.,!]+)',
        r'weather\s+for\s+([^?.,!]+)',
        r'weather\s+at\s+([^?.,!]+)',
        r'(?:in|at|for)\s+([a-zA-Z\s]+)(?:\s*weather|\s*$)',
        r'weather.*?([a-zA-Z]{3,}(?:\s+[a-zA-Z]{2,})*)\s*[?.]?\s*$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            location = match.group(1).strip()
            location = re.sub(r'\b(today|tomorrow|this\s+week|next\s+week|like|please|now|weather)\b', '', location).strip()
            location = ' '.join(location.split())
            
            if location and len(location) > 2:
                return ' '.join(word.capitalize() for word in location.split())
    
    return None

def send_progress_update(message: str):
    """Send progress update to client"""
    socketio.emit('message', {
        'type': 'system',
        'content': message,
        'timestamp': datetime.now().isoformat()
    })
    time.sleep(0.5)

def analyze_location_with_progress(location_input: str) -> str:
    """Analyze location with progress updates"""
    
    try:
        logger.info(f"Starting analysis for location: {location_input}")
        send_progress_update(f'Looking up location: {location_input}')
        
        lat, lon, standardized_name = location_service.get_coordinates_and_name(location_input)
        logger.info(f"Geocoding result: lat={lat}, lon={lon}, name={standardized_name}")
        
        if lat is None or lon is None:
            logger.warning(f"Could not geocode location: {location_input}")
            return f"Location Not Found: Could not find location '{location_input}'. Try being more specific (e.g., 'Miami, FL' instead of 'Miami')."
        
        send_progress_update(f'Found {standardized_name} ({lat:.4f}, {lon:.4f})')
        
        if rag_system:
            send_progress_update('Performing comprehensive weather analysis...')
            send_progress_update('This may take 30-60 seconds...')
            
            try:
                logger.info(f"Running RAG analysis for {standardized_name}")
                analysis = rag_system.analyze_location(lat, lon, standardized_name)
                logger.info("RAG analysis completed successfully")
                return format_analysis_response(analysis)
            except Exception as e:
                logger.error(f"RAG analysis failed: {e}")
                logger.error(traceback.format_exc())
                send_progress_update('AI analysis failed, using enhanced analysis...')
                return enhanced_basic_analysis(lat, lon, standardized_name)
        else:
            logger.info("Using enhanced basic analysis (no RAG system)")
            send_progress_update('Performing weather pattern analysis...')
            return enhanced_basic_analysis(lat, lon, standardized_name)
            
    except Exception as e:
        logger.error(f"Error in location analysis: {e}")
        logger.error(traceback.format_exc())
        return f"Analysis Failed: Analysis failed for '{location_input}': {str(e)}"

def enhanced_basic_analysis(lat: float, lon: float, location_name: str) -> str:
    """Enhanced basic analysis when AI is unavailable"""
    
    try:
        send_progress_update('Retrieving current weather forecast...')
        
        forecast_data = weather_collector.get_current_forecast(lat, lon, days=7)
        
        if not forecast_data:
            return f"Data Unavailable: Could not retrieve weather data for {location_name}."
        
        send_progress_update('Analyzing weather patterns...')
        
        end_date = datetime(2024, 12, 31).date()
        start_date = end_date - timedelta(days=14)
        
        historical_data = weather_collector.get_historical_data(
            lat, lon, start_date.isoformat(), end_date.isoformat()
        )
        
        send_progress_update('Generating analysis...')
        
        analyzer = EnhancedWeatherAnalyzer()
        
        # Generate the four required sections
        current_conditions = analyzer.generate_current_conditions(forecast_data, location_name)
        forecast_summary = analyzer.generate_7day_forecast_summary(forecast_data, location_name)
        extreme_predictions = analyzer.predict_extreme_weather_events(forecast_data, historical_data, location_name)
        intelligence_summary = analyzer.generate_weather_intelligence_summary(forecast_data, historical_data, location_name)
        
        # Combine sections
        response = f"{current_conditions}\n\n{forecast_summary}\n\n{extreme_predictions}\n\n{intelligence_summary}"
        
        response += f"\n\nAnalysis Details:\n"
        response += f"Location: {location_name} ({lat:.4f}, {lon:.4f})\n"
        response += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced basic analysis error: {e}")
        return f"Analysis Error: Weather analysis failed for {location_name}: {str(e)}"

def format_analysis_response(analysis: dict) -> str:
    """Format analysis response from RAG system"""
    
    try:
        location = analysis.get('location', {})
        location_name = location.get('name', 'Unknown Location')
        
        ai_analysis = analysis.get('ai_analysis', '')
        
        if ai_analysis and 'Unable to generate analysis' not in ai_analysis:
            return ai_analysis
        else:
            # Fallback to enhanced analysis
            predictions = analysis.get('predictions', [])
            historical_analysis = analysis.get('historical_analysis', {})
            
            response = f"WEATHER ANALYSIS - {location_name.upper()}\n\n"
            
            if predictions:
                high_risk = [p for p in predictions if p['risk_score'] >= 5]
                if high_risk:
                    response += "Immediate Weather Threats:\n"
                    for pred in high_risk[:3]:
                        response += f"• {pred['event_type']} (Risk: {pred['risk_score']}/10)\n"
                        response += f"  Time: {pred['timestamp']}\n"
                        response += f"  Conditions: {pred['conditions']['temperature']:.1f}°C, {pred['conditions']['wind_speed']:.1f} km/h winds\n\n"
                else:
                    response += "No immediate high-risk weather events detected\n\n"
            
            stats = historical_analysis.get('statistics', {})
            if stats:
                response += "Historical Weather Context:\n"
                response += f"• Recent extreme events: {stats.get('total_extreme_events', 0)}\n"
                response += f"• Peak wind speeds: {stats.get('max_wind_speed', 0):.1f} km/h\n"
                response += f"• Temperature range: {stats.get('min_temperature', 0):.1f}°C to {stats.get('max_temperature', 0):.1f}°C\n\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return f"Format Error: Error formatting analysis for {analysis.get('location', {}).get('name', 'location')}: {str(e)}"

def initialize_rag_system():
    """Initialize RAG system with enhanced response generation"""
    
    global rag_system
    try:
        api_key = os.getenv('WATSONX_API_KEY', WATSONX_API_KEY)
        project_id = os.getenv('WATSONX_PROJECT_ID', WATSONX_PROJECT_ID)
        endpoint = os.getenv('WATSONX_ENDPOINT', WATSONX_ENDPOINT)
        
        logger.info("Initializing enhanced RAG system")
        logger.info(f"API Key configured: {bool(api_key and api_key != 'your_api_key_here')}")
        logger.info(f"Project ID configured: {bool(project_id and project_id != 'your_project_id_here')}")
        
        if api_key == 'your_api_key_here' or project_id == 'your_project_id_here':
            logger.warning("Please update your watsonx.ai credentials")
            rag_system = ExtremeWeatherRAGSystem(
                watsonx_api_key="placeholder",
                watsonx_project_id="placeholder",
                watsonx_endpoint=endpoint
            )
            rag_system = enhance_rag_system_responses(rag_system)
            return False
            
        rag_system = ExtremeWeatherRAGSystem(
            watsonx_api_key=api_key,
            watsonx_project_id=project_id,
            watsonx_endpoint=endpoint
        )
        
        # Apply enhancements
        rag_system = enhance_rag_system_responses(rag_system)
        
        test_result = rag_system.quick_analysis(25.7617, -80.1918, "Miami, FL")
        if test_result and not test_result.startswith("Unable"):
            logger.info("Enhanced RAG system initialized and tested successfully")
            return True
        else:
            logger.warning("Enhanced RAG system initialized but test failed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize enhanced RAG system: {e}")
        try:
            rag_system = ExtremeWeatherRAGSystem("placeholder", "placeholder")
            rag_system = enhance_rag_system_responses(rag_system)
            return False
        except:
            return False

@app.route('/')
def index():
    """Main chat interface page"""
    return render_template('index.html', locations=DEFAULT_LOCATIONS)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    weather_api_working = test_weather_api()
    rag_ready = rag_system is not None
    
    return jsonify({
        'status': 'healthy',
        'rag_system_ready': rag_ready,
        'weather_api_ready': weather_api_working,
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'ai_analysis': 'Available' if rag_ready else 'Basic mode only',
            'global_coverage': True,
            'data_sources': ['Open-Meteo API', 'IBM watsonx.ai' if rag_ready else 'Local analysis']
        }
    })

@app.route('/api/locations')
def get_locations():
    """Get predefined locations"""
    return jsonify(DEFAULT_LOCATIONS)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('message', {
        'type': 'system',
        'content': """Welcome to the Extreme Weather RAG System

I can analyze weather patterns and predict extreme conditions worldwide.

Try these commands:
• /analyze [location] - Full weather risk analysis
• /predict [lat lon] - Coordinate-based prediction
• /help - Show all available commands

Or ask naturally:
• "Hurricane risk for Miami"
• "Heat wave predictions for Phoenix"
• "Show me tornado warnings for Oklahoma"

System Status: """ + ("AI Enhanced" if rag_system else "Basic Mode"),
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('message')
def handle_message(data):
    """Handle incoming messages"""
    try:
        user_message = data.get('message', '').strip()
        logger.info(f"Processing message: {user_message}")
        
        if not user_message:
            return
        
        emit('message', {
            'type': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            response = process_user_input(user_message)
            logger.info(f"Generated response length: {len(response)} characters")
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            logger.error(traceback.format_exc())
            response = f"Processing Error: Sorry, I encountered an error processing your request: {str(e)}"
        
        emit('message', {
            'type': 'bot',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        logger.error(traceback.format_exc())
        emit('message', {
            'type': 'error',
            'content': f'System Error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

def process_user_input(user_input: str) -> str:
    """Process user input and generate response"""
    user_input = user_input.strip()
    
    if user_input.startswith('/'):
        return handle_command(user_input)
    
    return handle_natural_query(user_input)

def handle_command(command: str) -> str:
    """Handle command-based input"""
    parts = command.split()
    cmd = parts[0].lower()
    
    try:
        if cmd == '/help':
            return """Weather RAG System Commands

Weather Analysis:
• /analyze [location] - Comprehensive weather risk analysis
• /predict [location or lat lon] - Detailed predictions
• /quick [location] - Fast weather summary

Location Examples:
• /analyze Tokyo, Japan
• /predict Miami, FL
• /predict 25.7617 -80.1918 (coordinates)

Information:
• /locations - Show high-risk locations
• /status - System status and capabilities
• /about - About this system

Natural Language:
You can also ask naturally:
• "Hurricane risk for Florida?"
• "Heat wave predictions for Phoenix"
• "What's the weather like in Sydney?"

Tips:
• Be specific with locations (e.g., "Miami, FL" not just "Miami")
• Use coordinates for precise analysis
• Commands work worldwide with any location"""

        elif cmd == '/status':
            weather_working = test_weather_api()
            return f"""System Status Report

AI Analysis: {"IBM watsonx.ai Ready" if rag_system else "Basic Mode (No AI credentials)"}
Weather Data: {"Open-Meteo API Connected" if weather_working else "Connection Issues"}
Location Service: Global coverage enabled
Data Sources: Open-Meteo (90TB+ historical), {"IBM Granite models" if rag_system else "Statistical analysis"}

Coverage:
• Historical data: 1940-present
• Forecast range: 16 days
• Global locations: Any coordinates
• Update frequency: Hourly

Capabilities:
{"• Full AI-powered analysis and predictions" if rag_system else "• Enhanced weather pattern analysis"}
• Extended weather event detection
• Long-term seasonal predictions
• Real-time monitoring
• Natural language processing

Performance:
• Response time: {"30-60 seconds (AI analysis)" if rag_system else "5-15 seconds (enhanced)"}
• Accuracy: {"High (AI-enhanced)" if rag_system else "Good (pattern-based)"}

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        elif cmd == '/locations':
            response = "High-Risk Weather Locations\n\n"
            for i, loc in enumerate(DEFAULT_LOCATIONS, 1):
                response += f"{i}. {loc['name']}\n"
                response += f"   Coordinates: {loc['lat']}, {loc['lon']}\n"
                response += f"   Risk Profile: {loc['description']}\n"
                response += f"   Try: /analyze {loc['name']}\n\n"
            
            response += "Global Coverage:\n"
            response += "• Any city name (e.g., /analyze Berlin, Germany)\n"
            response += "• Coordinates (e.g., /predict 52.5200 13.4050)\n"
            response += "• Natural language (e.g., \"Hurricane risk for Caribbean\")"
            return response

        elif cmd == '/about':
            return f"""Extreme Weather RAG System

Mission: Save lives through AI-powered early warning systems

Technology Stack:
• AI Engine: {"IBM watsonx.ai Granite models" if rag_system else "Enhanced statistical algorithms"}
• Weather Data: Open-Meteo API (90+ TB historical data)
• Coverage: Global (any location worldwide)
• Analysis: RAG (Retrieval-Augmented Generation)

Capabilities:
• Hurricane/Typhoon prediction and tracking
• Tornado risk assessment and warnings
• Heat wave and cold wave detection
• Severe thunderstorm analysis
• Flash flood prediction
• Long-term seasonal forecasting

Hackathon Project:
Built for IBM watsonx.ai Hackathon - Climate Challenge
Team: CodeX (Nihal, Eza, Manisha, Zakaria)

Data Sources:
• Historical weather: 1940-present (Open-Meteo Archive)
• Real-time forecasts: 16-day predictions
• {"AI insights: IBM watsonx.ai" if rag_system else "Enhanced analysis: Local algorithms"}

Impact:
Protecting communities worldwide through intelligent weather analysis."""

        elif cmd == '/analyze':
            if len(parts) < 2:
                return "Missing Location: Please specify a location.\n\nExamples:\n• /analyze Tokyo, Japan\n• /analyze Miami, FL\n• /analyze Phoenix, Arizona"
            
            location_input = ' '.join(parts[1:])
            return analyze_location_with_progress(location_input)

        elif cmd == '/predict':
            if len(parts) >= 3:
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return predict_coordinates(lat, lon)
                    else:
                        return "Invalid Coordinates: Latitude must be -90 to 90, longitude -180 to 180."
                except ValueError:
                    pass
            
            if len(parts) >= 2:
                location_input = ' '.join(parts[1:])
                return analyze_location_with_progress(location_input)
            else:
                return "Missing Input: Please provide location or coordinates.\n\nExamples:\n• /predict Tokyo, Japan\n• /predict 35.6762 139.6503"

        elif cmd == '/quick':
            if len(parts) < 2:
                return "Missing Location: Please specify a location.\n\nExample: /quick London, UK"
            
            location_input = ' '.join(parts[1:])
            return quick_weather_summary(location_input)

        else:
            return f"Unknown Command: Unknown command: {cmd}\n\nType /help to see all available commands."

    except Exception as e:
        logger.error(f"Error in command handling: {e}")
        return f"Command Error: Error processing command {cmd}: {str(e)}"

def handle_natural_query(query: str) -> str:
    """Handle natural language queries"""
    
    logger.info(f"Processing natural language query: {query}")
    
    location = parse_location_from_text(query)
    logger.info(f"Extracted location: {location}")
    
    if location:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['quick', 'brief', 'summary']):
            logger.info(f"Processing quick summary for: {location}")
            return quick_weather_summary(location)
        else:
            logger.info(f"Processing full analysis for: {location}")
            return analyze_location_with_progress(location)
    
    logger.warning(f"No location found in query: {query}")
    return """I'd be happy to help with weather analysis!

Please specify a location in your query:

Examples:
• "Hurricane risk for Miami"
• "What's the weather like in Tokyo?"
• "Heat wave predictions for Phoenix"
• "Tornado warnings for Oklahoma City"

Quick Commands:
• /analyze [location] - Full analysis
• /quick [location] - Fast summary
• /help - See all commands

Supported Locations:
• Any city worldwide
• Coordinates (latitude, longitude)
• Regions (e.g., "Gulf Coast")"""

def predict_coordinates(lat: float, lon: float) -> str:
    """Predict weather for specific coordinates"""
    
    try:
        send_progress_update(f'Analyzing coordinates {lat:.4f}, {lon:.4f}')
        
        location_name = get_location_name(lat, lon)
        send_progress_update(f'Identified: {location_name}')
        
        if rag_system:
            send_progress_update('Running comprehensive analysis...')
            analysis = rag_system.analyze_location(lat, lon, location_name)
            return format_analysis_response(analysis)
        else:
            return enhanced_basic_analysis(lat, lon, location_name)
            
    except Exception as e:
        logger.error(f"Error in coordinate prediction: {e}")
        return f"Coordinate Analysis Failed: Error analyzing coordinates {lat}, {lon}: {str(e)}"

def get_location_name(lat: float, lon: float) -> str:
    """Get location name from coordinates"""
    
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1,
            "zoom": 10
        }
        headers = {"User-Agent": "WeatherRAGSystem/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            parts = []
            if address.get('city'):
                parts.append(address['city'])
            elif address.get('town'):
                parts.append(address['town'])
            elif address.get('village'):
                parts.append(address['village'])
            elif address.get('county'):
                parts.append(address['county'])
                
            if address.get('country_code') == 'us' and address.get('state'):
                parts.append(address['state'])
            elif address.get('country'):
                parts.append(address['country'])
            
            if parts:
                return ', '.join(parts)
    except:
        pass
    
    # Fallback to descriptive location
    ns = "Northern" if lat >= 0 else "Southern"
    ew = "Eastern" if lon >= 0 else "Western"
    
    if abs(lat) < 23.5:
        region = "Tropical"
    elif abs(lat) < 40:
        region = "Subtropical"
    elif abs(lat) < 60:
        region = "Temperate"
    else:
        region = "Polar"
        
    return f"{region} {ns} {ew} Region ({lat:.3f}, {lon:.3f})"

def quick_weather_summary(location_input: str) -> str:
    """Generate quick weather summary"""
    
    try:
        lat, lon, name = location_service.get_coordinates_and_name(location_input)
        
        if lat is None:
            return f"Location Not Found: Could not find location: {location_input}"
        
        send_progress_update(f'Generating weather summary for {name}')
        
        forecast_data = weather_collector.get_current_forecast(lat, lon, days=3)
        if not forecast_data:
            return f"Data Unavailable: Weather data unavailable for {name}"
        
        analyzer = EnhancedWeatherAnalyzer()
        
        current_conditions = analyzer.generate_current_conditions(forecast_data, name)
        
        return f"{name} - Weather Summary\n\n{current_conditions}"
        
    except Exception as e:
        return f"Quick Summary Failed: {str(e)}"

def test_weather_api() -> bool:
    """Test weather API connectivity"""
    try:
        response = weather_collector.get_current_forecast(40.7128, -74.0060, days=1)
        return bool(response and response.get('hourly'))
    except:
        return False

if __name__ == '__main__':
    # Initialize logging directory
    os.makedirs('logs', exist_ok=True)
    
    print("\n" + "="*60)
    print("EXTREME WEATHER RAG SYSTEM - CLEAN VERSION")
    print("="*60)
    
    rag_initialized = initialize_rag_system()
    weather_working = test_weather_api()
    
    print(f"RAG System: {'Ready' if rag_initialized else 'Basic Mode'}")
    print(f"Weather API: {'Connected' if weather_working else 'Issues detected'}")
    print(f"Global Coverage: Available")
    print(f"Web Interface: Starting on http://localhost:{FLASK_PORT}")
    print("="*60)
    
    if not rag_initialized:
        print("\nTo enable full AI features:")
        print("   1. Set WATSONX_API_KEY environment variable")
        print("   2. Set WATSONX_PROJECT_ID environment variable")
        print("   3. Or update credentials in src/config.py")
        print("\n   System will run in enhanced mode!")
    
    if not weather_working:
        print("\nWeather API connectivity issues detected")
        print("   System will use mock data for demonstration")
    
    print(f"\nStarting server on http://localhost:{FLASK_PORT}")
    print("   Press Ctrl+C to stop\n")
    
    # Start the Flask-SocketIO server
    socketio.run(
        app, 
        host=FLASK_HOST, 
        port=FLASK_PORT, 
        debug=FLASK_DEBUG,
        allow_unsafe_werkzeug=True
    )
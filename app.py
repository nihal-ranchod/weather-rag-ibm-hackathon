#!/usr/bin/env python3
"""
Weather RAG Chat Interface
Flask web application with real-time chat for extreme weather predictions

Author: Climate RAG Team
Date: May 2025
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import json
import logging
import os
import sys
from datetime import datetime
import traceback
import requests
import re
from typing import Optional, Tuple

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from weather_rag_system import ExtremeWeatherRAGSystem, WeatherDataCollector, ExtremeWeatherDetector
    from config import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you've created the weather_rag_system.py and config.py files")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'weather_rag_secret_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
rag_system = None
weather_collector = WeatherDataCollector()
detector = ExtremeWeatherDetector()

class LocationService:
    """Handles all location-related operations"""
    
    def __init__(self):
        self.cache = {}  # Cache for performance
        
    def get_coordinates_and_name(self, location_input: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Get coordinates and standardized name for any location input
        Returns: (lat, lon, standardized_name)
        """
        
        # Check cache first
        cache_key = location_input.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        lat, lon, name = None, None, None
        
        # Try multiple geocoding services
        for service in [self._try_nominatim, self._try_photon]:
            try:
                lat, lon, name = service(location_input)
                if lat is not None and lon is not None:
                    # Cache the result
                    self.cache[cache_key] = (lat, lon, name)
                    return lat, lon, name
            except Exception as e:
                logger.debug(f"Geocoding service failed: {e}")
                continue
        
        return None, None, None
    
    def _try_nominatim(self, location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try OpenStreetMap Nominatim (free, no API key)"""
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "accept-language": "en"  # Force English results
        }
        headers = {"User-Agent": "WeatherRAGSystem/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Build standardized English name
                address = result.get('address', {})
                name_parts = []
                
                # Prefer original query for well-known cities
                query_lower = location.lower()
                if any(city in query_lower for city in ['tokyo', 'sydney', 'paris', 'london', 'miami']):
                    # Use a cleaned version of the original query
                    if 'tokyo' in query_lower:
                        name_parts = ['Tokyo', 'Japan']
                    elif 'sydney' in query_lower:
                        name_parts = ['Sydney', 'Australia']
                    elif 'paris' in query_lower:
                        name_parts = ['Paris', 'France']
                    elif 'london' in query_lower:
                        name_parts = ['London', 'UK']
                    elif 'miami' in query_lower:
                        name_parts = ['Miami', 'FL']
                else:
                    # Build from address components
                    city = (address.get('city') or address.get('town') or 
                        address.get('village') or address.get('municipality'))
                    if city:
                        name_parts.append(city)
                    
                    # State/Province and Country
                    if address.get('country_code') == 'us' and address.get('state'):
                        name_parts.append(address['state'])
                    elif address.get('country') and address.get('country') != city:
                        name_parts.append(address['country'])
                
                name = ', '.join(name_parts) if name_parts else result.get('display_name', location)
                return lat, lon, name
        
        return None, None, None
    
    def _try_photon(self, location: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try Photon geocoding (based on OpenStreetMap)"""
        url = "https://photon.komoot.io/api/"
        params = {"q": location, "limit": 1}
        
        response = requests.get(url, params=params, timeout=5)
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
                
                name = ', '.join(name_parts) if name_parts else location
                return lat, lon, name
        
        return None, None, None

def parse_location_from_text(text: str) -> Optional[str]:
    """Extract location from natural language using patterns"""
    text_lower = text.lower()
    
    # Direct city mentions (case-insensitive)
    direct_cities = {
        'reykjavik': 'Reykjavik, Iceland',
        'tokyo': 'Tokyo, Japan',
        'sydney': 'Sydney, Australia',
        'paris': 'Paris, France',
        'london': 'London, UK',
        'miami': 'Miami, FL',
        'new york': 'New York, NY',
        'los angeles': 'Los Angeles, CA'
    }
    
    # Check for direct city mentions first
    for city, full_name in direct_cities.items():
        if city in text_lower:
            return full_name
    
    # Pattern 1: "weather [preposition] [location]"
    patterns = [
        r'(?:weather|conditions|forecast)\s+(?:for|in|at)\s+([^?.,!]+)',
        r'(?:analyze|check)\s+([^?.,!]+)',
        r'(?:hurricane|tornado|storm)\s+(?:risk|threat)\s+(?:for|in)\s+([^?.,!]+)',
        r'(?:what\'s|how\'s)\s+(?:the\s+)?weather\s+(?:like\s+)?(?:in|at)\s+([^?.,!]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            location = match.group(1).strip()
            # Clean up common words
            location = re.sub(r'\b(today|tomorrow|this week|next week|like)\b', '', location).strip()
            if location and len(location) > 2:
                return location.title()
    
    return None

def analyze_location_smart(location_input: str) -> str:
    """Smart location analysis using geocoding service"""
    
    # Initialize location service
    if not hasattr(analyze_location_smart, 'location_service'):
        analyze_location_smart.location_service = LocationService()
    
    location_service = analyze_location_smart.location_service
    
    # Get coordinates and standardized name
    lat, lon, standardized_name = location_service.get_coordinates_and_name(location_input)
    
    if lat is None or lon is None:
        return f"❌ Could not find location '{location_input}'. Please try a more specific location or use coordinates with `/predict [lat] [lon]`"
    
    # Show what we found
    socketio.emit('message', {
        'type': 'system',
        'content': f'🔍 Found: {standardized_name} ({lat:.4f}, {lon:.4f})\nAnalyzing weather patterns...',
        'timestamp': datetime.now().isoformat()
    })
    
    if not rag_system:
        return basic_weather_analysis(lat, lon, standardized_name)
    
    try:
        analysis = rag_system.analyze_location(lat, lon, standardized_name)
        return format_analysis_response(analysis)
    except Exception as e:
        logger.error(f"Error in smart location analysis: {e}")
        return f"❌ Error analyzing {standardized_name}: {str(e)}"

def initialize_rag_system():
    """Initialize the RAG system with credentials"""
    global rag_system
    try:
        # Load credentials from environment or config
        api_key = os.getenv('WATSONX_API_KEY', WATSONX_API_KEY)
        project_id = os.getenv('WATSONX_PROJECT_ID', WATSONX_PROJECT_ID)
        endpoint = os.getenv('WATSONX_ENDPOINT', WATSONX_ENDPOINT)
        
        if api_key == 'your_api_key_here' or project_id == 'your_project_id_here':
            logger.warning("⚠️  Please update your watsonx.ai credentials in .env or config.py")
            return False
            
        rag_system = ExtremeWeatherRAGSystem(
            watsonx_api_key=api_key,
            watsonx_project_id=project_id,
            watsonx_endpoint=endpoint
        )
        logger.info("✅ RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface page"""
    return render_template('index.html', locations=DEFAULT_LOCATIONS)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_system_ready': rag_system is not None,
        'timestamp': datetime.now().isoformat()
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
        'content': 'Welcome to the Extreme Weather RAG System! \n\n' +
                  'I can help you analyze extreme weather patterns and predict dangerous conditions.\n\n' +
                  'Try these commands:\n' +
                  '• /analyze [location] - Analyze weather risks\n' +
                  '• /predict [lat] [lon] - Predict extreme weather\n' +
                  '• /monitor [location] - Start monitoring\n' +
                  '• /locations - Show predefined locations\n' +
                  '• /help - Show all commands\n\n' +
                  'Or just ask me about weather conditions!',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('message')
def handle_message(data):
    """Handle incoming chat messages"""
    try:
        user_message = data.get('message', '').strip()
        logger.info(f"Received message: {user_message}")
        
        if not user_message:
            return
        
        # Echo user message
        emit('message', {
            'type': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process command or query
        response = process_user_input(user_message)
        
        # Send bot response
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
            'content': f'Sorry, I encountered an error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

def process_user_input(user_input):
    """Process user input and return appropriate response"""
    user_input = user_input.strip()
    
    # Handle commands
    if user_input.startswith('/'):
        return handle_command(user_input)
    
    # Handle natural language queries
    return handle_natural_query(user_input)

def handle_command(command):
    """Enhanced command handling with smart location parsing"""
    parts = command.split()
    cmd = parts[0].lower()
    
    if cmd == '/help':
        return """🆘 **Available Commands:**

**Weather Analysis:**
- `/analyze [location]` - Comprehensive weather risk analysis
- `/predict [location or lat lon]` - Coordinate or location-based prediction  
- `/monitor [location]` - Start continuous monitoring
- `/quick [location]` - Quick weather summary

**Information:**
- `/locations` - Show predefined high-risk locations
- `/about` - About this system
- `/status` - System status

**Examples:**
- `/analyze Tokyo, Japan`
- `/predict Miami, FL` or `/predict 25.7617 -80.1918`
- `/monitor Sydney, Australia`
- `/quick London, UK`

You can also ask natural questions like:
- "What's the hurricane risk for Florida this week?"
- "Are there any tornado warnings for Oklahoma?"
- "Show me heat wave predictions for Phoenix"
"""
    
    elif cmd == '/locations':
        response = "📍 **Predefined High-Risk Locations:**\n\n"
        for i, loc in enumerate(DEFAULT_LOCATIONS, 1):
            response += f"{i}. **{loc['name']}** ({loc['lat']}, {loc['lon']})\n"
            response += f"   Risk Profile: {loc['description']}\n\n"
        response += "Use `/analyze [location]` to get detailed analysis for any location worldwide."
        return response
    
    elif cmd == '/status':
        return f"""🔧 **System Status:**

- RAG System: {'✅ Ready' if rag_system else '❌ Not initialized'}
- Weather API: {'✅ Connected' if test_weather_api() else '❌ Connection issues'}
- AI Analysis: {'✅ Available' if rag_system else '❌ Credentials needed'}
- Location Service: ✅ Global coverage enabled

**Data Sources:**
- Historical Weather: Open-Meteo Archive API (1940-present)
- AI Analysis: IBM watsonx.ai Granite models
- Geocoding: OpenStreetMap Nominatim + Photon
- Coverage: Global (any location worldwide)

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    elif cmd == '/about':
        return """🌪️ **Extreme Weather RAG System**

This AI-powered system combines:
- **90+ TB of historical weather data** (1940-present)
- **Real-time monitoring** and prediction
- **IBM watsonx.ai** for intelligent analysis
- **Global location recognition** via smart geocoding
- **Community-focused alerts** in plain language

**Capabilities:**
- Hurricane/Typhoon prediction
- Tornado risk assessment  
- Heat wave/cold wave detection
- Severe thunderstorm analysis
- Flash flood warnings
- Blizzard and ice storm alerts

**Built for:** IBM watsonx.ai Hackathon - Climate Challenge
**Purpose:** Save lives through AI-powered early warning systems
**Coverage:** Global (any location worldwide)
"""
    
    elif cmd == '/analyze':
        if len(parts) < 2:
            return "❌ Please specify a location. Example: `/analyze Tokyo, Japan`"
        
        # Join all parts after command as location
        location_input = ' '.join(parts[1:])
        return analyze_location_smart(location_input)
    
    elif cmd == '/predict':
        if len(parts) >= 3:
            # Try coordinates first
            try:
                lat = float(parts[1])
                lon = float(parts[2])
                return predict_coordinates_command(lat, lon)
            except ValueError:
                pass
        
        if len(parts) >= 2:
            # Location name provided
            location_input = ' '.join(parts[1:])
            return analyze_location_smart(location_input)
        else:
            return "❌ Provide coordinates or location. Examples: `/predict 40.7128 -74.0060` or `/predict Tokyo, Japan`"
    
    elif cmd == '/monitor':
        if len(parts) < 2:
            return "❌ Please specify a location. Example: `/monitor Sydney, Australia`"
        
        location_input = ' '.join(parts[1:])
        return start_monitoring_command(location_input)
    
    elif cmd == '/quick':
        if len(parts) < 2:
            return "❌ Please specify a location. Example: `/quick London, UK`"
        
        location_input = ' '.join(parts[1:])
        return quick_summary_command(location_input)
    
    else:
        return f"❌ Unknown command: {cmd}\nType `/help` for available commands."

def handle_natural_query(query):
    """Enhanced natural language processing"""
    
    # Extract location from query
    location = parse_location_from_text(query)
    
    if location:
        query_lower = query.lower()
        
        # Determine analysis type based on keywords
        if any(word in query_lower for word in ['hurricane', 'typhoon', 'cyclone']):
            focus = 'hurricane'
        elif any(word in query_lower for word in ['tornado', 'twister']):
            focus = 'tornado'
        elif any(word in query_lower for word in ['heat', 'hot', 'temperature']):
            focus = 'heat'
        elif any(word in query_lower for word in ['flood', 'rain', 'precipitation']):
            focus = 'flood'
        elif any(word in query_lower for word in ['snow', 'blizzard', 'winter']):
            focus = 'winter'
        else:
            focus = None
        
        return analyze_location_smart(location)
    
    # If no location found, provide helpful guidance
    return """🤔 I'd be happy to help analyze weather conditions! 

Please specify a location in your query. For example:
- "Hurricane risk for Miami"
- "What's the weather like in Tokyo?"
- "Analyze Sydney, Australia"
- "Tornado warnings for Oklahoma City"

Or use specific commands:
- `/analyze [location]` for comprehensive analysis
- `/predict [location or coordinates]` for predictions
- `/help` to see all available commands
"""

def analyze_location_command(location, focus=None):
    """Handle location analysis command"""
    try:
        # Try to find coordinates for location
        coords = find_coordinates_for_location(location)
        if not coords:
            return f"❌ Could not find coordinates for '{location}'. Please try a more specific location or use coordinates with `/predict [lat] [lon]`"
        
        lat, lon = coords
        
        if not rag_system:
            # Fallback to basic analysis without AI
            return basic_weather_analysis(lat, lon, location, focus)
        
        # Full RAG analysis
        socketio.emit('message', {
            'type': 'system',
            'content': f'🔍 Analyzing extreme weather patterns for {location}...\nThis may take 30-60 seconds.',
            'timestamp': datetime.now().isoformat()
        })
        
        analysis = rag_system.analyze_location(lat, lon, location, days_history=30)
        
        return format_analysis_response(analysis, focus)
        
    except Exception as e:
        logger.error(f"Error in location analysis: {e}")
        return f"❌ Error analyzing {location}: {str(e)}"

def predict_coordinates_command(lat, lon):
    """Handle coordinate prediction command with location lookup"""
    try:
        # Try to get a meaningful location name from coordinates
        location_name = get_location_name_from_coordinates(lat, lon)
        
        if not rag_system:
            return basic_weather_analysis(lat, lon, location_name)
        
        socketio.emit('message', {
            'type': 'system', 
            'content': f'🔍 Analyzing coordinates {lat}, {lon} ({location_name})...',
            'timestamp': datetime.now().isoformat()
        })
        
        analysis = rag_system.analyze_location(lat, lon, location_name)
        return format_analysis_response(analysis)
        
    except Exception as e:
        logger.error(f"Error in coordinate prediction: {e}")
        return f"❌ Error analyzing coordinates: {str(e)}"

def get_location_name_from_coordinates(lat, lon):
    """Try to get a location name from coordinates with multiple services"""
    
    # Method 1: Check known locations first (fastest)
    known_locations = {
        (25.7617, -80.1918): "Miami, FL",
        (40.7128, -74.0060): "New York, NY", 
        (34.0522, -118.2437): "Los Angeles, CA",
        (41.8781, -87.6298): "Chicago, IL",
        (29.7604, -95.3698): "Houston, TX",
        (33.4484, -112.0740): "Phoenix, AZ",
        (39.7392, -104.9903): "Denver, CO",
        (47.6062, -122.3321): "Seattle, WA",
        (35.4676, -97.5164): "Oklahoma City, OK",
        (42.8864, -78.8784): "Buffalo, NY",
        (35.3395, -97.4864): "Moore, OK",
        (29.9511, -90.0715): "New Orleans, LA",
        (48.8566, 2.3522): "Paris, France",
        (-33.8688, 151.2093): "Sydney, Australia",
        (35.6762, 139.6503): "Tokyo, Japan",
        (55.7558, 37.6176): "Moscow, Russia",
        (19.4326, -99.1332): "Mexico City, Mexico",
        (51.5074, -0.1278): "London, UK"
    }
    
    coord_key = (round(lat, 4), round(lon, 4))
    if coord_key in known_locations:
        return known_locations[coord_key]
    
    # Check for approximate matches (within 0.01 degrees)
    for (known_lat, known_lon), name in known_locations.items():
        if abs(lat - known_lat) < 0.01 and abs(lon - known_lon) < 0.01:
            return name
    
    # Method 2: Try BigDataCloud (free, no API key)
    try:
        url = f"https://api.bigdatacloud.net/data/reverse-geocode-client"
        params = {
            "latitude": lat,
            "longitude": lon,
            "localityLanguage": "en"
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Build location name from available data
            parts = []
            if data.get('city'):
                parts.append(data['city'])
            elif data.get('locality'):
                parts.append(data['locality'])
            elif data.get('principalSubdivision'):
                parts.append(data['principalSubdivision'])
                
            if data.get('countryCode'):
                if data['countryCode'] == 'US' and data.get('principalSubdivisionCode'):
                    parts.append(data['principalSubdivisionCode'])
                else:
                    parts.append(data['countryCode'])
            
            if parts:
                return ', '.join(parts)
                
    except Exception as e:
        logger.error(f"BigDataCloud geocoding failed: {e}")
    
    # Method 3: Try Nominatim (OpenStreetMap - free, no API key)
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1
        }
        headers = {"User-Agent": "WeatherRAGSystem/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            # Build location name
            parts = []
            if address.get('city'):
                parts.append(address['city'])
            elif address.get('town'):
                parts.append(address['town'])
            elif address.get('village'):
                parts.append(address['village'])
            elif address.get('county'):
                parts.append(address['county'])
                
            if address.get('country_code'):
                if address['country_code'] == 'us' and address.get('state'):
                    parts.append(address['state'])
                else:
                    parts.append(address['country_code'].upper())
            
            if parts:
                return ', '.join(parts)
                
    except Exception as e:
        logger.error(f"Nominatim geocoding failed: {e}")
    
    # Method 4: Descriptive fallback based on coordinates
    def describe_location(lat, lon):
        # Hemisphere descriptions
        ns = "Northern" if lat >= 0 else "Southern"
        ew = "Eastern" if lon >= 0 else "Western"
        
        # Rough regional descriptions
        if abs(lat) < 23.5:
            region = "Tropical"
        elif abs(lat) < 40:
            region = "Subtropical" 
        elif abs(lat) < 60:
            region = "Temperate"
        else:
            region = "Polar"
            
        return f"{region} {ns} {ew} Region ({lat:.3f}, {lon:.3f})"
    
    return describe_location(lat, lon)

def start_monitoring_command(location):
    """Handle monitoring command"""
    coords = find_coordinates_for_location(location)
    if not coords:
        return f"❌ Could not find coordinates for '{location}'"
    
    lat, lon = coords
    
    # In a real implementation, this would start background monitoring
    return f"""🔄 **Monitoring Started for {location}**

📍 Coordinates: {lat}, {lon}
⏰ Check Interval: Every hour
🚨 Alert Threshold: Risk level 6/10 or higher

**What I'm monitoring:**
• Hurricane/tropical storm formation
• Tornado-favorable conditions  
• Severe thunderstorm development
• Extreme temperature events
• Heavy precipitation/flood risks

**Note:** This is a demo system. In production, monitoring would run continuously in the background and send real-time alerts via SMS, email, or emergency broadcast systems.

Use `/analyze {location}` to get a current detailed assessment.
"""

def quick_summary_command(location):
    """Handle quick summary command"""
    coords = find_coordinates_for_location(location)
    if not coords:
        return f"❌ Could not find coordinates for '{location}'"
    
    lat, lon = coords
    
    try:
        # Get current forecast
        forecast_data = weather_collector.get_current_forecast(lat, lon, days=3)
        
        if not forecast_data:
            return f"❌ Could not retrieve weather data for {location}"
        
        # Quick analysis
        predictions = detector.predict_extreme_events(forecast_data, {})
        
        response = f"⚡ **Quick Weather Summary - {location}**\n\n"
        
        if predictions:
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            if high_risk:
                response += "🚨 **HIGH RISK EVENTS DETECTED:**\n"
                for pred in high_risk[:3]:
                    response += f"• {pred['event_type']} - Risk {pred['risk_score']}/10\n"
                    response += f"  Time: {pred['timestamp']}\n"
                    response += f"  Factors: {', '.join(pred['risk_factors'][:2])}\n\n"
            else:
                response += "✅ No high-risk weather events detected in next 3 days\n\n"
        else:
            response += "✅ No extreme weather threats detected\n\n"
        
        # Current conditions
        if 'hourly' in forecast_data and forecast_data['hourly']:
            current = forecast_data['hourly']
            response += "🌤️ **Current Conditions:**\n"
            response += f"Temperature: {current.get('temperature_2m', [0])[0]}°C\n"
            response += f"Wind Speed: {current.get('wind_speed_10m', [0])[0]} km/h\n"
            response += f"Precipitation: {current.get('precipitation', [0])[0]} mm\n"
        
        response += f"\nFor detailed analysis, use: `/analyze {location}`"
        return response
        
    except Exception as e:
        return f"❌ Error getting summary: {str(e)}"

def find_coordinates_for_location(location):
    """Find coordinates for a location name"""
    # Simple mapping for common locations
    location_coords = {
        'miami': (25.7617, -80.1918),
        'miami, fl': (25.7617, -80.1918),
        'moore': (35.3395, -97.4864),
        'moore, ok': (35.3395, -97.4864),
        'oklahoma city': (35.4676, -97.5164),
        'oklahoma city, ok': (35.4676, -97.5164),
        'phoenix': (33.4484, -112.0740),
        'phoenix, az': (33.4484, -112.0740),
        'buffalo': (42.8864, -78.8784),
        'buffalo, ny': (42.8864, -78.8784),
        'new orleans': (29.9511, -90.0715),
        'new orleans, la': (29.9511, -90.0715),
        'houston': (29.7604, -95.3698),
        'houston, tx': (29.7604, -95.3698),
        'los angeles': (34.0522, -118.2437),
        'los angeles, ca': (34.0522, -118.2437),
        'chicago': (41.8781, -87.6298),
        'chicago, il': (41.8781, -87.6298),
        'denver': (39.7392, -104.9903),
        'denver, co': (39.7392, -104.9903),
        'atlanta': (33.7490, -84.3880),
        'atlanta, ga': (33.7490, -84.3880)
    }
    
    location_lower = location.lower().strip()
    return location_coords.get(location_lower)

def basic_weather_analysis(lat, lon, location_name, focus=None):
    """Basic weather analysis without AI (fallback)"""
    try:
        # Get weather data
        forecast_data = weather_collector.get_current_forecast(lat, lon, days=7)
        historical_data = weather_collector.get_historical_data(
            lat, lon, 
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # Analyze patterns
        historical_analysis = detector.analyze_historical_patterns(historical_data)
        predictions = detector.predict_extreme_events(forecast_data, historical_analysis)
        
        response = f"📊 **Weather Analysis - {location_name}**\n\n"
        
        # Historical summary
        stats = historical_analysis.get('statistics', {})
        response += "📈 **30-Day Historical Summary:**\n"
        response += f"• Extreme events detected: {stats.get('total_extreme_events', 0)}\n"
        response += f"• Max wind speed: {stats.get('max_wind_speed', 0):.1f} km/h\n"
        response += f"• Temperature range: {stats.get('min_temperature', 0):.1f}°C to {stats.get('max_temperature', 0):.1f}°C\n"
        response += f"• Total precipitation: {stats.get('total_precipitation', 0):.1f} mm\n\n"
        
        # Predictions
        if predictions:
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            if high_risk:
                response += "🚨 **HIGH RISK PREDICTIONS (Next 7 Days):**\n"
                for pred in high_risk:
                    response += f"• **{pred['event_type']}** - Risk Level {pred['risk_score']}/10\n"
                    response += f"  📅 Time: {pred['timestamp']}\n"
                    response += f"  ⚠️  Factors: {', '.join(pred['risk_factors'])}\n"
                    response += f"  🌡️ Conditions: {pred['conditions']['temperature']:.1f}°C, "
                    response += f"{pred['conditions']['wind_speed']:.1f} km/h winds\n\n"
            else:
                response += "✅ **No high-risk weather events predicted for next 7 days**\n\n"
        
        # Focus-specific analysis
        if focus == 'hurricane':
            response += "🌀 **Hurricane Risk Assessment:**\n"
            hurricane_risk = any(p['risk_score'] >= 6 and 'wind' in str(p['risk_factors']).lower() for p in predictions)
            response += f"Current Risk: {'HIGH' if hurricane_risk else 'LOW'}\n"
        elif focus == 'tornado':
            response += "🌪️ **Tornado Risk Assessment:**\n"
            tornado_risk = any('thunderstorm' in str(p['risk_factors']).lower() for p in predictions)
            response += f"Current Risk: {'ELEVATED' if tornado_risk else 'LOW'}\n"
        elif focus == 'heat':
            response += "🌡️ **Heat Wave Risk Assessment:**\n"
            heat_risk = any('heat' in str(p['risk_factors']).lower() for p in predictions)
            response += f"Current Risk: {'HIGH' if heat_risk else 'MODERATE'}\n"
        
        response += "\n⚠️ **Note:** This is basic analysis. For AI-powered insights, please configure watsonx.ai credentials."
        
        return response
        
    except Exception as e:
        return f"❌ Error in weather analysis: {str(e)}"

def format_analysis_response(analysis, focus=None):
    """Format comprehensive analysis response"""
    try:
        location = analysis.get('location', {})
        location_name = location.get('name', 'Unknown Location')
        
        response = f"🌍 **Comprehensive Weather Analysis - {location_name}**\n\n"
        
        # Historical analysis
        hist_analysis = analysis.get('historical_analysis', {})
        stats = hist_analysis.get('statistics', {})
        
        response += "📊 **Historical Patterns (30 days):**\n"
        response += f"• Extreme events: {stats.get('total_extreme_events', 0)}\n"
        response += f"• Max wind speed: {stats.get('max_wind_speed', 0):.1f} km/h\n"
        response += f"• Temperature range: {stats.get('min_temperature', 0):.1f}°C to {stats.get('max_temperature', 0):.1f}°C\n"
        response += f"• Total precipitation: {stats.get('total_precipitation', 0):.1f} mm\n\n"
        
        # Predictions
        predictions = analysis.get('predictions', [])
        if predictions:
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            if high_risk:
                response += "🚨 **HIGH RISK PREDICTIONS:**\n"
                for pred in high_risk[:3]:  # Show top 3
                    response += f"• **{pred['event_type']}** (Risk: {pred['risk_score']}/10)\n"
                    response += f"  📅 {pred['timestamp']}\n"
                    response += f"  ⚠️  {', '.join(pred['risk_factors'][:2])}\n\n"
        
        # AI Analysis - SHOW FULL RESPONSE (no truncation)
        ai_analysis = analysis.get('ai_analysis', '')
        if ai_analysis and ai_analysis != "Unable to generate analysis at this time.":
            response += "🤖 **AI-Powered Insights:**\n"
            response += ai_analysis  # REMOVED truncation here!
            response += "\n\n"
        
        # Community alerts
        alerts = analysis.get('community_alerts', [])
        if alerts:
            response += "📢 **Community Alerts:**\n"
            for alert in alerts[:2]:  # Show top 2 alerts
                response += f"🚨 Risk Level {alert['risk_score']}/10\n"
                response += alert['alert_text']  # REMOVED truncation here too!
                response += "\n\n"
        
        # Historical insights - SHOW FULL RESPONSE
        hist_insights = analysis.get('historical_insights', '')
        if hist_insights and 'No significant' not in hist_insights:
            response += "📈 **Historical Climate Patterns:**\n"
            response += hist_insights  # REMOVED truncation here!
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return f"❌ Error formatting analysis: {str(e)}"

def test_weather_api():
    """Test weather API connectivity"""
    try:
        response = weather_collector.get_current_forecast(40.7128, -74.0060, days=1)
        return bool(response)
    except:
        return False

if __name__ == '__main__':
    # Initialize logging directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize RAG system
    rag_initialized = initialize_rag_system()
    
    if not rag_initialized:
        logger.warning("⚠️  RAG system not initialized. Basic weather analysis will be available.")
        logger.warning("⚠️  Please update your watsonx.ai credentials to enable full AI features.")
    
    # Test weather API
    if test_weather_api():
        logger.info("✅ Weather API connection successful")
    else:
        logger.warning("⚠️  Weather API connection issues")
    
    print("\n" + "="*60)
    print("🌪️  EXTREME WEATHER RAG SYSTEM")
    print("="*60)
    print(f"🔧 RAG System: {'✅ Ready' if rag_initialized else '❌ Credentials needed'}")
    print(f"🌐 Weather API: {'✅ Connected' if test_weather_api() else '❌ Connection issues'}")
    print(f"💻 Web Interface: Starting on http://localhost:{FLASK_PORT}")
    print("="*60)
    
    if not rag_initialized:
        print("\n⚠️  To enable full AI features:")
        print("   1. Update credentials in .env file")
        print("   2. Or set environment variables:")
        print("      export WATSONX_API_KEY='your_key'")
        print("      export WATSONX_PROJECT_ID='your_project_id'")
        print("\n   Basic weather analysis is still available!")
    
    print(f"\n🚀 Starting server on http://localhost:{FLASK_PORT}")
    print("   Press Ctrl+C to stop\n")
    
    # Start the Flask-SocketIO server
    socketio.run(
        app, 
        host=FLASK_HOST, 
        port=FLASK_PORT, 
        debug=FLASK_DEBUG,
        allow_unsafe_werkzeug=True
    )
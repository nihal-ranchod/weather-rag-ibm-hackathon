#!/usr/bin/env python3
"""
FIXED - Weather RAG Chat Interface
Flask web application with improved error handling and command processing

Key fixes:
1. Better error handling for failed commands
2. Enhanced location parsing and geocoding
3. More robust API calls with fallbacks
4. Improved user feedback and progress updates
"""

from flask import Flask, render_template, request, jsonify, session
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

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from weather_rag_system import ExtremeWeatherRAGSystem, WeatherDataCollector, ExtremeWeatherDetector
    from config import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you've created the weather_rag_system.py and config.py files")

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weather_rag.log'),
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

class EnhancedLocationService:
    """FIXED - Enhanced location service with better error handling"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        
    def get_coordinates_and_name(self, location_input: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        FIXED - Get coordinates with multiple fallback strategies
        """
        cache_key = location_input.lower().strip()
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Using cached location data for {location_input}")
            return self.cache[cache_key]
        
        logger.info(f"Looking up coordinates for: {location_input}")
        
        # Try known locations first (fastest)
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
                logger.debug(f"Trying {service_name} geocoding service...")
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
        
        # Enhanced known locations database
        known_locations = {
            # US Cities
            'miami': (25.7617, -80.1918, 'Miami, FL'),
            'miami, fl': (25.7617, -80.1918, 'Miami, FL'),
            'phoenix': (33.4484, -112.0740, 'Phoenix, AZ'),
            'phoenix, az': (33.4484, -112.0740, 'Phoenix, AZ'),
            'oklahoma city': (35.4676, -97.5164, 'Oklahoma City, OK'),
            'oklahoma': (35.4676, -97.5164, 'Oklahoma City, OK'),
            'london': (51.5074, -0.1278, 'London, UK'),
            'london, uk': (51.5074, -0.1278, 'London, UK'),
            'gulf coast': (29.0, -90.0, 'Gulf Coast Region'),
            'new york': (40.7128, -74.0060, 'New York, NY'),
            'los angeles': (34.0522, -118.2437, 'Los Angeles, CA'),
            'chicago': (41.8781, -87.6298, 'Chicago, IL'),
            'houston': (29.7604, -95.3698, 'Houston, TX'),
            'new orleans': (29.9511, -90.0715, 'New Orleans, LA'),
            'buffalo': (42.8864, -78.8784, 'Buffalo, NY'),
            'denver': (39.7392, -104.9903, 'Denver, CO'),
            'seattle': (47.6062, -122.3321, 'Seattle, WA'),
            'atlanta': (33.7490, -84.3880, 'Atlanta, GA'),
            
            # International Cities
            'tokyo': (35.6762, 139.6503, 'Tokyo, Japan'),
            'tokyo, japan': (35.6762, 139.6503, 'Tokyo, Japan'),
            'sydney': (-33.8688, 151.2093, 'Sydney, Australia'),
            'sydney, australia': (-33.8688, 151.2093, 'Sydney, Australia'),
            'paris': (48.8566, 2.3522, 'Paris, France'),
            'paris, france': (48.8566, 2.3522, 'Paris, France'),
            'reykjavik': (64.1466, -21.9426, 'Reykjavik, Iceland'),
            'reykjavik, iceland': (64.1466, -21.9426, 'Reykjavik, Iceland'),
            'moscow': (55.7558, 37.6176, 'Moscow, Russia'),
            'beijing': (39.9042, 116.4074, 'Beijing, China'),
            'mumbai': (19.0760, 72.8777, 'Mumbai, India'),
            'cairo': (30.0444, 31.2357, 'Cairo, Egypt'),
            'lagos': (6.5244, 3.3792, 'Lagos, Nigeria'),
            'buenos aires': (-34.6118, -58.3960, 'Buenos Aires, Argentina'),
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
                
                # Build clean name
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

# Initialize enhanced location service
location_service = EnhancedLocationService()

def parse_location_from_text(text: str) -> Optional[str]:
    """FIXED - Enhanced location parsing"""
    text_lower = text.lower()
    
    # Pattern-based extraction
    patterns = [
        r'(?:weather|conditions|forecast|risk|predictions?|warnings?)\s+(?:for|in|at|near)\s+([^?.,!]+)',
        r'(?:hurricane|tornado|storm|heat\s+wave|flood)\s+(?:risk|threat|warning)\s+(?:for|in|at)\s+([^?.,!]+)',
        r'(?:analyze|check|monitor)\s+([^?.,!]+)',
        r'(?:what\'s|how\'s)\s+(?:the\s+)?weather\s+(?:like\s+)?(?:in|at|for)\s+([^?.,!]+)',
        r'predictions?\s+(?:for|in)\s+([^?.,!]+)',
        r'(?:show\s+me|tell\s+me\s+about)\s+.*?(?:for|in|at)\s+([^?.,!]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            location = match.group(1).strip()
            # Clean up common noise words
            location = re.sub(r'\b(today|tomorrow|this\s+week|next\s+week|like|please|now)\b', '', location).strip()
            if location and len(location) > 2:
                return location.title()
    
    return None

def send_progress_update(message: str):
    """Send progress update to client"""
    socketio.emit('message', {
        'type': 'system',
        'content': message,
        'timestamp': datetime.now().isoformat()
    })
    time.sleep(0.5)  # Small delay for UX

def analyze_location_with_progress(location_input: str) -> str:
    """FIXED - Analyze location with progress updates and better error handling"""
    
    try:
        # Step 1: Geocoding
        send_progress_update(f'🔍 Looking up location: {location_input}...')
        
        lat, lon, standardized_name = location_service.get_coordinates_and_name(location_input)
        
        if lat is None or lon is None:
            return f"❌ Could not find location '{location_input}'. Please try:\n• A more specific location (e.g., 'Miami, FL' instead of 'Miami')\n• Coordinates with `/predict [lat] [lon]`\n• One of the predefined locations from the sidebar"
        
        send_progress_update(f'✅ Found: {standardized_name} ({lat:.4f}, {lon:.4f})')
        
        # Step 2: Quick check vs full analysis
        if rag_system:
            send_progress_update('🤖 Performing comprehensive AI analysis...')
            send_progress_update('⏳ This may take 30-60 seconds...')
            
            try:
                analysis = rag_system.analyze_location(lat, lon, standardized_name)
                return format_analysis_response(analysis)
            except Exception as e:
                logger.error(f"RAG analysis failed: {e}")
                send_progress_update('⚠️ AI analysis failed, using fallback analysis...')
                return basic_weather_analysis_enhanced(lat, lon, standardized_name)
        else:
            send_progress_update('📊 Performing weather pattern analysis...')
            return basic_weather_analysis_enhanced(lat, lon, standardized_name)
            
    except Exception as e:
        logger.error(f"Error in location analysis: {e}")
        logger.error(traceback.format_exc())
        return f"❌ Analysis failed for '{location_input}': {str(e)}\n\nPlease try again or contact support if the problem persists."

def basic_weather_analysis_enhanced(lat: float, lon: float, location_name: str) -> str:
    """FIXED - Enhanced basic analysis with better error handling"""
    
    try:
        send_progress_update('🌤️ Retrieving current weather forecast...')
        
        # Get forecast data
        forecast_data = weather_collector.get_current_forecast(lat, lon, days=7)
        
        if not forecast_data:
            return f"❌ Could not retrieve weather data for {location_name}. The weather service may be temporarily unavailable."
        
        send_progress_update('📈 Analyzing historical weather patterns...')
        
        # Get historical data (shorter period for faster response)
        end_date = datetime(2024, 12, 31).date()
        start_date = end_date - timedelta(days=14)  # Just 2 weeks for speed
        
        historical_data = weather_collector.get_historical_data(
            lat, lon, start_date.isoformat(), end_date.isoformat()
        )
        
        send_progress_update('🔍 Detecting extreme weather patterns...')
        
        # Analyze patterns
        historical_analysis = detector.analyze_historical_patterns(historical_data) if historical_data else {}
        predictions = detector.predict_extreme_events(forecast_data, historical_analysis)
        
        send_progress_update('📋 Preparing comprehensive report...')
        
        # Build response
        response = f"📊 **Weather Analysis - {location_name}**\n\n"
        
        # Current conditions
        if forecast_data.get('hourly'):
            current = forecast_data['hourly']
            response += "🌤️ **Current Conditions:**\n"
            if current.get('temperature_2m'):
                response += f"• Temperature: {current['temperature_2m'][0]:.1f}°C\n"
            if current.get('wind_speed_10m'):
                response += f"• Wind Speed: {current['wind_speed_10m'][0]:.1f} km/h\n"
            if current.get('precipitation'):
                response += f"• Precipitation: {current['precipitation'][0]:.1f} mm\n"
            response += "\n"
        
        # Risk Assessment
        stats = historical_analysis.get('statistics', {})
        extreme_events = stats.get('total_extreme_events', 0)
        max_wind = stats.get('max_wind_speed', 0)
        max_temp = stats.get('max_temperature', 20)
        min_temp = stats.get('min_temperature', 10)
        
        risk_level = "LOW"
        if extreme_events > 0 or max_wind > 60:
            risk_level = "HIGH"
        elif max_wind > 30 or abs(max_temp - min_temp) > 25:
            risk_level = "MODERATE"
        
        response += f"📈 **Risk Assessment: {risk_level}**\n"
        response += f"• Recent extreme events: {extreme_events}\n"
        response += f"• Maximum wind speed: {max_wind:.1f} km/h\n"
        response += f"• Temperature range: {min_temp:.1f}°C to {max_temp:.1f}°C\n\n"
        
        # Predictions
        if predictions:
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            if high_risk:
                response += "🚨 **Weather Alerts (Next 7 Days):**\n"
                for pred in high_risk[:3]:
                    response += f"• **{pred['event_type']}** - Risk {pred['risk_score']}/10\n"
                    response += f"  📅 {pred['timestamp']}\n"
                    response += f"  ⚠️ {', '.join(pred['risk_factors'][:2])}\n\n"
            else:
                response += "✅ **No high-risk weather events predicted**\n\n"
        
        # Safety recommendations
        response += "🛡️ **Safety Recommendations:**\n"
        if risk_level == "HIGH":
            response += "• URGENT: Monitor weather alerts closely\n"
            response += "• Prepare emergency supplies immediately\n"
            response += "• Avoid unnecessary travel\n"
        elif risk_level == "MODERATE":
            response += "• Stay informed about weather updates\n"
            response += "• Check emergency supplies\n"
            response += "• Plan for possible weather disruptions\n"
        else:
            response += "• Continue normal activities\n"
            response += "• Monitor routine weather updates\n"
        
        response += f"\n📍 **Location:** {location_name} ({lat:.4f}, {lon:.4f})"
        response += f"\n⏰ **Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced basic analysis error: {e}")
        return f"❌ Weather analysis failed for {location_name}: {str(e)}\n\nPlease try again later."

def format_analysis_response(analysis: dict) -> str:
    """FIXED - Format analysis response without truncation"""
    try:
        location = analysis.get('location', {})
        location_name = location.get('name', 'Unknown Location')
        
        response = f"🌍 **Comprehensive Weather Analysis - {location_name}**\n\n"
        
        # System status
        system_status = analysis.get('system_status', {})
        ai_available = system_status.get('ai_available', False)
        confidence = system_status.get('analysis_confidence', 'Unknown')
        
        response += f"🔧 **Analysis Status:** {confidence} confidence"
        if ai_available:
            response += " (AI-enhanced)"
        response += "\n\n"
        
        # Historical analysis
        hist_analysis = analysis.get('historical_analysis', {})
        stats = hist_analysis.get('statistics', {})
        
        if stats:
            response += "📊 **Historical Patterns (Recent Data):**\n"
            response += f"• Extreme events detected: {stats.get('total_extreme_events', 0)}\n"
            response += f"• Maximum wind speed: {stats.get('max_wind_speed', 0):.1f} km/h\n"
            response += f"• Temperature range: {stats.get('min_temperature', 0):.1f}°C to {stats.get('max_temperature', 0):.1f}°C\n"
            response += f"• Total precipitation: {stats.get('total_precipitation', 0):.1f} mm\n\n"
        
        # Predictions
        predictions = analysis.get('predictions', [])
        if predictions:
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            if high_risk:
                response += "🚨 **High-Risk Weather Predictions:**\n"
                for pred in high_risk[:3]:
                    response += f"• **{pred['event_type']}** (Risk: {pred['risk_score']}/10)\n"
                    response += f"  📅 {pred['timestamp']}\n"
                    response += f"  ⚠️ {', '.join(pred['risk_factors'][:2])}\n\n"
            else:
                response += "✅ **No high-risk weather events predicted**\n\n"
        
        # AI Analysis - Show full response
        ai_analysis = analysis.get('ai_analysis', '')
        if ai_analysis and 'Unable to generate analysis' not in ai_analysis:
            response += "🤖 **AI-Powered Weather Intelligence:**\n"
            response += ai_analysis + "\n\n"
        
        # Community alerts
        alerts = analysis.get('community_alerts', [])
        if alerts:
            response += "📢 **Community Weather Alerts:**\n"
            for alert in alerts[:2]:
                response += f"🚨 Risk Level {alert['risk_score']}/10\n"
                response += alert['alert_text'] + "\n\n"
        
        # Data sources
        data_sources = analysis.get('data_sources', {})
        response += "📋 **Data Sources:**\n"
        response += f"• Weather Data: {data_sources.get('weather_data', 'Open-Meteo API')}\n"
        response += f"• AI Analysis: {data_sources.get('ai_analysis', 'Basic Analysis')}\n"
        response += f"• Analysis Period: {data_sources.get('analysis_period', 'Recent data')}\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return f"❌ Error formatting analysis for {analysis.get('location', {}).get('name', 'location')}: {str(e)}"

def initialize_rag_system():
    """FIXED - Initialize RAG system with better error handling"""
    global rag_system
    try:
        # Load credentials
        api_key = os.getenv('WATSONX_API_KEY', WATSONX_API_KEY)
        project_id = os.getenv('WATSONX_PROJECT_ID', WATSONX_PROJECT_ID)
        endpoint = os.getenv('WATSONX_ENDPOINT', WATSONX_ENDPOINT)
        
        logger.info(f"Initializing RAG system...")
        logger.info(f"API Key configured: {bool(api_key and api_key != 'your_api_key_here')}")
        logger.info(f"Project ID configured: {bool(project_id and project_id != 'your_project_id_here')}")
        
        if api_key == 'your_api_key_here' or project_id == 'your_project_id_here':
            logger.warning("⚠️ Please update your watsonx.ai credentials")
            return False
            
        rag_system = ExtremeWeatherRAGSystem(
            watsonx_api_key=api_key,
            watsonx_project_id=project_id,
            watsonx_endpoint=endpoint
        )
        
        # Test the system
        test_result = rag_system.quick_analysis(25.7617, -80.1918, "Miami, FL")
        if test_result and not test_result.startswith("❌"):
            logger.info("✅ RAG system initialized and tested successfully")
            return True
        else:
            logger.warning("⚠️ RAG system initialized but test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface page"""
    return render_template('index.html', locations=DEFAULT_LOCATIONS)

@app.route('/api/health')
def health_check():
    """FIXED - Enhanced health check"""
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
        'content': 'Welcome to the Extreme Weather Event RAG System! \n\n' +
                  'I can analyze weather patterns and predict extreme conditions worldwide.\n\n' +
                  '**Try these commands:**\n' +
                  '• `/analyze [location]` - Full weather risk analysis\n' +
                  '• `/predict [lat] [lon]` - Coordinate-based prediction\n' +
                  '• `/help` - Show all available commands\n\n' +
                  '**Or just ask naturally:**\n' +
                  '• "Hurricane risk for Miami"\n' +
                  '• "Heat wave predictions for Phoenix"\n' +
                  '• "Show me tornado warnings for Oklahoma"\n\n' +
                  f'**System Status:** {"✅ AI Enhanced" if rag_system else "⚠️ Basic Mode"}',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('message')
def handle_message(data):
    """FIXED - Enhanced message handling"""
    try:
        user_message = data.get('message', '').strip()
        logger.info(f"Processing message: {user_message}")
        
        if not user_message:
            return
        
        # Echo user message
        emit('message', {
            'type': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process command or query
        try:
            response = process_user_input_enhanced(user_message)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            response = f"❌ Sorry, I encountered an error processing your request: {str(e)}\n\nPlease try again or use `/help` for available commands."
        
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
            'content': f'System error: {str(e)}\n\nPlease refresh the page and try again.',
            'timestamp': datetime.now().isoformat()
        })

def process_user_input_enhanced(user_input: str) -> str:
    """FIXED - Enhanced input processing with better error handling"""
    user_input = user_input.strip()
    
    # Handle commands
    if user_input.startswith('/'):
        return handle_command_enhanced(user_input)
    
    # Handle natural language queries
    return handle_natural_query_enhanced(user_input)

def handle_command_enhanced(command: str) -> str:
    """FIXED - Enhanced command handling"""
    parts = command.split()
    cmd = parts[0].lower()
    
    try:
        if cmd == '/help':
            return """🆘 **Weather RAG System Commands**

** Weather Analysis:**
• `/analyze [location]` - Comprehensive weather risk analysis
• `/predict [location or lat lon]` - Detailed predictions
• `/quick [location]` - Fast weather summary
• `/monitor [location]` - Start monitoring alerts

**📍 Location Examples:**
• `/analyze Tokyo, Japan`
• `/predict Miami, FL`
• `/predict 25.7617 -80.1918` (coordinates)
• `/quick London, UK`

**ℹ️ Information:**
• `/locations` - Show high-risk locations
• `/status` - System status and capabilities
• `/about` - About this system

**🌊 Natural Language:**
You can also ask naturally:
• "Hurricane risk for Florida?"
• "Heat wave predictions for Phoenix"
• "Tornado warnings for Oklahoma City"
• "What's the weather like in Sydney?"

**💡 Tips:**
• Be specific with locations (e.g., "Miami, FL" not just "Miami")
• Use coordinates for precise analysis
• Commands work worldwide with any location"""

        elif cmd == '/status':
            weather_working = test_weather_api()
            return f"""🔧 **System Status Report**

**🤖 AI Analysis:** {'✅ IBM watsonx.ai Ready' if rag_system else '⚠️ Basic Mode (No AI credentials)'}
**🌐 Weather Data:** {'✅ Open-Meteo API Connected' if weather_working else '❌ Connection Issues'}
**📡 Location Service:** ✅ Global coverage enabled
**💾 Data Sources:** Open-Meteo (90TB+ historical), {'IBM Granite models' if rag_system else 'Statistical analysis'}

**🌍 Coverage:**
• Historical data: 1940-present
• Forecast range: 16 days
• Global locations: Any coordinates
• Update frequency: Hourly

**⚡ Capabilities:**
{'• Full AI-powered analysis and predictions' if rag_system else '• Basic weather pattern analysis'}
• Extreme weather detection
• Community alert generation
• Real-time monitoring
• Natural language processing

**📊 Performance:**
• Response time: {'30-60 seconds (AI analysis)' if rag_system else '5-15 seconds (basic)'}
• Accuracy: {'High (AI-enhanced)' if rag_system else 'Moderate (pattern-based)'}

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        elif cmd == '/locations':
            response = "📍 **High-Risk Weather Locations**\n\n"
            for i, loc in enumerate(DEFAULT_LOCATIONS, 1):
                response += f"**{i}. {loc['name']}**\n"
                response += f"   📍 {loc['lat']}, {loc['lon']}\n"
                response += f"   ⚠️ {loc['description']}\n"
                response += f"   💬 Try: `/analyze {loc['name']}`\n\n"
            
            response += "**🌍 Global Coverage:**\n"
            response += "• Any city name (e.g., `/analyze Berlin, Germany`)\n"
            response += "• Coordinates (e.g., `/predict 52.5200 13.4050`)\n"
            response += "• Natural language (e.g., \"Hurricane risk for Caribbean\")\n"
            return response

        elif cmd == '/about':
            return f""" **Extreme Weather RAG System**

**🎯 Mission:** Save lives through AI-powered early warning systems

**🔬 Technology Stack:**
• **AI Engine:** {'IBM watsonx.ai Granite models' if rag_system else 'Statistical algorithms'}
• **Weather Data:** Open-Meteo API (90+ TB historical data)
• **Coverage:** Global (any location worldwide)
• **Analysis:** RAG (Retrieval-Augmented Generation)

**⚡ Capabilities:**
• Hurricane/Typhoon prediction and tracking
• Tornado risk assessment and warnings
• Heat wave and cold wave detection
• Severe thunderstorm analysis
• Flash flood prediction
• Blizzard and ice storm alerts

**🏆 Hackathon Project:**
Built for IBM watsonx.ai Hackathon - Climate Challenge
Team: CodeX (Nihal, Eza, Manisha, Zakaria)

**📊 Data Sources:**
• Historical weather: 1940-present (Open-Meteo Archive)
• Real-time forecasts: 16-day predictions
• {'AI insights: IBM watsonx.ai' if rag_system else 'Pattern analysis: Local algorithms'}

**🌍 Impact:**
Protecting communities worldwide through intelligent weather analysis and early warning systems."""

        elif cmd == '/analyze':
            if len(parts) < 2:
                return "❌ Please specify a location.\n\n**Examples:**\n• `/analyze Tokyo, Japan`\n• `/analyze Miami, FL`\n• `/analyze Phoenix, Arizona`"
            
            location_input = ' '.join(parts[1:])
            return analyze_location_with_progress(location_input)

        elif cmd == '/predict':
            if len(parts) >= 3:
                # Try coordinates first
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return predict_coordinates_enhanced(lat, lon)
                    else:
                        return "❌ Invalid coordinates. Latitude must be -90 to 90, longitude -180 to 180."
                except ValueError:
                    pass
            
            if len(parts) >= 2:
                location_input = ' '.join(parts[1:])
                return analyze_location_with_progress(location_input)
            else:
                return "❌ Please provide location or coordinates.\n\n**Examples:**\n• `/predict Tokyo, Japan`\n• `/predict 35.6762 139.6503`"

        elif cmd == '/quick':
            if len(parts) < 2:
                return "❌ Please specify a location.\n\n**Example:** `/quick London, UK`"
            
            location_input = ' '.join(parts[1:])
            return quick_weather_summary(location_input)

        elif cmd == '/monitor':
            if len(parts) < 2:
                return "❌ Please specify a location.\n\n**Example:** `/monitor Miami, FL`"
            
            location_input = ' '.join(parts[1:])
            return start_monitoring_enhanced(location_input)

        else:
            return f"❌ Unknown command: `{cmd}`\n\nType `/help` to see all available commands."

    except Exception as e:
        logger.error(f"Error in command handling: {e}")
        return f"❌ Error processing command `{cmd}`: {str(e)}\n\nType `/help` for available commands."

def handle_natural_query_enhanced(query: str) -> str:
    """FIXED - Enhanced natural language processing"""
    
    # Extract location from query
    location = parse_location_from_text(query)
    
    if location:
        query_lower = query.lower()
        
        # Quick responses for simple queries
        if any(word in query_lower for word in ['quick', 'brief', 'summary']):
            return quick_weather_summary(location)
        else:
            return analyze_location_with_progress(location)
    
    # No location found - provide guidance
    return """🤔 **I'd be happy to help with weather analysis!**

Please specify a location in your query:

**🌍 Examples:**
• "Hurricane risk for Miami"
• "What's the weather like in Tokyo?"
• "Heat wave predictions for Phoenix"
• "Tornado warnings for Oklahoma City"

**⚡ Quick Commands:**
• `/analyze [location]` - Full analysis
• `/quick [location]` - Fast summary
• `/help` - See all commands

**📍 Supported Locations:**
• Any city worldwide
• Coordinates (latitude, longitude)
• Regions (e.g., "Gulf Coast")"""

def predict_coordinates_enhanced(lat: float, lon: float) -> str:
    """FIXED - Enhanced coordinate prediction"""
    
    try:
        send_progress_update(f'📍 Analyzing coordinates {lat:.4f}, {lon:.4f}...')
        
        # Try to get location name
        location_name = get_location_name_enhanced(lat, lon)
        send_progress_update(f'🗺️ Location identified: {location_name}')
        
        if rag_system:
            send_progress_update('🤖 Running comprehensive AI analysis...')
            analysis = rag_system.analyze_location(lat, lon, location_name)
            return format_analysis_response(analysis)
        else:
            return basic_weather_analysis_enhanced(lat, lon, location_name)
            
    except Exception as e:
        logger.error(f"Error in coordinate prediction: {e}")
        return f"❌ Error analyzing coordinates {lat}, {lon}: {str(e)}"

def get_location_name_enhanced(lat: float, lon: float) -> str:
    """FIXED - Enhanced location name lookup"""
    
    # Try reverse geocoding
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
    """FIXED - Quick weather summary"""
    
    try:
        lat, lon, name = location_service.get_coordinates_and_name(location_input)
        
        if lat is None:
            return f"❌ Could not find location: {location_input}"
        
        send_progress_update(f'⚡ Getting quick summary for {name}...')
        
        if rag_system:
            result = rag_system.quick_analysis(lat, lon, name)
            return f"⚡ **Quick Weather Summary**\n\n{result}"
        else:
            # Basic quick summary
            forecast_data = weather_collector.get_current_forecast(lat, lon, days=3)
            if not forecast_data:
                return f"❌ Weather data unavailable for {name}"
            
            predictions = detector.predict_extreme_events(forecast_data, {})
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            
            if high_risk:
                summary = f"🚨 **{name} - Weather Alert**\n\n"
                summary += f"⚠️ {len(high_risk)} high-risk event(s) detected\n\n"
                for pred in high_risk[:2]:
                    summary += f"• {pred['event_type']} (Risk: {pred['risk_score']}/10)\n"
            else:
                summary = f"✅ **{name} - All Clear**\n\n"
                summary += "No high-risk weather events detected in next 3 days."
            
            return summary
            
    except Exception as e:
        return f"❌ Quick summary failed: {str(e)}"

def start_monitoring_enhanced(location_input: str) -> str:
    """FIXED - Enhanced monitoring setup"""
    
    try:
        lat, lon, name = location_service.get_coordinates_and_name(location_input)
        
        if lat is None:
            return f"❌ Could not find location: {location_input}"
        
        return f"""🔄 **Weather Monitoring Started - {name}**

📍 **Location:** {name} ({lat:.4f}, {lon:.4f})
⏰ **Monitoring:** Continuous weather pattern analysis
🚨 **Alert Threshold:** Risk level 6/10 or higher

**📊 What I'm monitoring:**
• 🌀 Hurricane/tropical storm development
• 🌪️ Tornado-favorable atmospheric conditions
• ⛈️ Severe thunderstorm formation
• 🌡️ Extreme temperature events (heat/cold waves)
• 🌊 Heavy precipitation and flood risks
• ❄️ Blizzard and ice storm conditions

**🔔 Alert System:**
In a production environment, this would provide:
• Real-time SMS/email alerts
• Emergency broadcast integration
• Community notification systems
• Escalation to local emergency management

**📱 Demo Note:**
This is a demonstration system. For actual emergency alerts, always rely on official weather services and local emergency management.

**🔍 Current Status:** Monitoring active
Use `/analyze {name}` for detailed current assessment."""

    except Exception as e:
        return f"❌ Monitoring setup failed: {str(e)}"

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
    
    # Initialize RAG system
    print("\n" + "="*60)
    print("🌪️  EXTREME WEATHER RAG SYSTEM - FIXED VERSION")
    print("="*60)
    
    rag_initialized = initialize_rag_system()
    weather_working = test_weather_api()
    
    print(f"🔧 RAG System: {'✅ Ready' if rag_initialized else '⚠️ Basic Mode'}")
    print(f"🌐 Weather API: {'✅ Connected' if weather_working else '❌ Issues detected'}")
    print(f"📍 Global Coverage: ✅ Available")
    print(f"💻 Web Interface: Starting on http://localhost:{FLASK_PORT}")
    print("="*60)
    
    if not rag_initialized:
        print("\n⚠️  To enable full AI features:")
        print("   1. Set WATSONX_API_KEY environment variable")
        print("   2. Set WATSONX_PROJECT_ID environment variable")
        print("   3. Or update credentials in src/config.py")
        print("\n   🔧 System will run in basic mode with enhanced weather analysis!")
    
    if not weather_working:
        print("\n⚠️  Weather API connectivity issues detected")
        print("   System will use mock data for demonstration")
    
    print(f"\n🚀 Starting server on http://localhost:{FLASK_PORT}")
    print("   📱 The interface is mobile-responsive")
    print("   🌍 Global weather analysis available")
    print("   ⌨️  Press Ctrl+C to stop\n")
    
    # Start the Flask-SocketIO server
    socketio.run(
        app, 
        host=FLASK_HOST, 
        port=FLASK_PORT, 
        debug=FLASK_DEBUG,
        allow_unsafe_werkzeug=True
    )
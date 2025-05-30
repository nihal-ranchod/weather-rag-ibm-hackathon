#!/usr/bin/env python3
"""
Extreme Weather Event Prediction RAG System
Built for IBM watsonx.ai Hackathon - Climate Challenge

This system combines:
1. Historical weather data retrieval from Open-Meteo API
2. Real-time weather monitoring 
3. RAG-based knowledge retrieval using watsonx.ai
4. Extreme weather event prediction and community alerts
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Collects weather data from Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.historical_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = f"{self.base_url}/forecast"
        
    def get_historical_data(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict:
        """
        Retrieve historical weather data for extreme weather analysis
        
        Args:
            lat: Latitude
            lon: Longitude  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary containing historical weather data
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
            "timezone": "auto"
        }
        
        try:
            response = requests.get(self.historical_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical data: {e}")
            return {}
    
    def get_current_forecast(self, lat: float, lon: float, days: int = 7) -> Dict:
        """
        Get current weather forecast for prediction analysis
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of forecast days (1-16)
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,weather_code,precipitation_probability",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,precipitation_probability_max",
            "forecast_days": days,
            "timezone": "auto"
        }
        
        try:
            response = requests.get(self.forecast_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching forecast data: {e}")
            return {}

class ExtremeWeatherDetector:
    """Analyzes weather data to detect extreme weather patterns"""
    
    def __init__(self):
        # Extreme weather thresholds (customizable by region)
        self.thresholds = {
            "hurricane_wind_speed": 119,  # km/h (74 mph)
            "severe_thunderstorm_wind": 93,  # km/h (58 mph)
            "heavy_rain_daily": 50,  # mm per day
            "extreme_rain_hourly": 25,  # mm per hour
            "heat_wave_temp": 35,  # Celsius (95°F)
            "cold_wave_temp": -18,  # Celsius (0°F)
            "blizzard_wind": 56,  # km/h (35 mph) with snow
            "tornado_conditions": {
                "wind_shear": 40,  # km/h difference
                "instability": 2500,  # J/kg CAPE approximation
                "moisture": 80  # relative humidity %
            }
        }
    
    def analyze_historical_patterns(self, weather_data: Dict) -> Dict:
        """
        Analyze historical data to identify extreme weather patterns
        
        Args:
            weather_data: Historical weather data from Open-Meteo
            
        Returns:
            Dictionary with extreme weather analysis
        """
        if not weather_data or 'hourly' not in weather_data:
            return {"error": "Invalid weather data"}
        
        hourly = weather_data['hourly']
        daily = weather_data.get('daily', {})
        
        # Convert to pandas for easier analysis
        df_hourly = pd.DataFrame(hourly)
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])
        
        extreme_events = []
        
        # Detect extreme wind events
        high_winds = df_hourly[df_hourly['wind_speed_10m'] > self.thresholds['severe_thunderstorm_wind']]
        if not high_winds.empty:
            for _, event in high_winds.iterrows():
                extreme_events.append({
                    "type": "High Wind Event",
                    "timestamp": event['time'].isoformat(),
                    "wind_speed": event['wind_speed_10m'],
                    "severity": "Hurricane" if event['wind_speed_10m'] > self.thresholds['hurricane_wind_speed'] else "Severe"
                })
        
        # Detect heavy precipitation events
        heavy_rain = df_hourly[df_hourly['precipitation'] > self.thresholds['extreme_rain_hourly']]
        if not heavy_rain.empty:
            for _, event in heavy_rain.iterrows():
                extreme_events.append({
                    "type": "Heavy Precipitation",
                    "timestamp": event['time'].isoformat(),
                    "precipitation": event['precipitation'],
                    "severity": "Extreme" if event['precipitation'] > 50 else "Heavy"
                })
        
        # Detect temperature extremes
        heat_events = df_hourly[df_hourly['temperature_2m'] > self.thresholds['heat_wave_temp']]
        cold_events = df_hourly[df_hourly['temperature_2m'] < self.thresholds['cold_wave_temp']]
        
        for _, event in heat_events.iterrows():
            extreme_events.append({
                "type": "Heat Wave",
                "timestamp": event['time'].isoformat(),
                "temperature": event['temperature_2m'],
                "severity": "Extreme" if event['temperature_2m'] > 40 else "High"
            })
        
        for _, event in cold_events.iterrows():
            extreme_events.append({
                "type": "Cold Wave", 
                "timestamp": event['time'].isoformat(),
                "temperature": event['temperature_2m'],
                "severity": "Extreme" if event['temperature_2m'] < -25 else "Severe"
            })
        
        # Calculate weather statistics
        stats = {
            "total_extreme_events": len(extreme_events),
            "avg_wind_speed": df_hourly['wind_speed_10m'].mean(),
            "max_wind_speed": df_hourly['wind_speed_10m'].max(),
            "total_precipitation": df_hourly['precipitation'].sum(),
            "avg_temperature": df_hourly['temperature_2m'].mean(),
            "max_temperature": df_hourly['temperature_2m'].max(),
            "min_temperature": df_hourly['temperature_2m'].min()
        }
        
        return {
            "extreme_events": extreme_events,
            "statistics": stats,
            "analysis_period": {
                "start": df_hourly['time'].min().isoformat(),
                "end": df_hourly['time'].max().isoformat()
            }
        }
    
    def predict_extreme_events(self, forecast_data: Dict, historical_analysis: Dict) -> List[Dict]:
        """
        Predict potential extreme weather events based on forecast and historical patterns
        
        Args:
            forecast_data: Current forecast data
            historical_analysis: Historical extreme weather analysis
            
        Returns:
            List of predicted extreme weather events
        """
        if not forecast_data or 'hourly' not in forecast_data:
            return []
        
        predictions = []
        hourly = forecast_data['hourly']
        df_forecast = pd.DataFrame(hourly)
        df_forecast['time'] = pd.to_datetime(df_forecast['time'])
        
        # Analyze upcoming conditions
        for i, row in df_forecast.iterrows():
            risk_factors = []
            risk_score = 0
            
            # High wind risk
            if row['wind_speed_10m'] > self.thresholds['severe_thunderstorm_wind']:
                risk_factors.append("High wind speeds predicted")
                risk_score += 3
                
            if row.get('wind_gusts_10m', 0) > self.thresholds['hurricane_wind_speed']:
                risk_factors.append("Hurricane-force wind gusts")
                risk_score += 4
            
            # Heavy precipitation risk
            if row['precipitation'] > self.thresholds['extreme_rain_hourly']:
                risk_factors.append("Heavy precipitation expected")
                risk_score += 3
            
            # Temperature extremes
            if row['temperature_2m'] > self.thresholds['heat_wave_temp']:
                risk_factors.append("Extreme heat conditions")
                risk_score += 2
            elif row['temperature_2m'] < self.thresholds['cold_wave_temp']:
                risk_factors.append("Extreme cold conditions")
                risk_score += 2
            
            # Severe thunderstorm conditions
            if (row['precipitation'] > 5 and 
                row['wind_speed_10m'] > 50 and 
                row['relative_humidity_2m'] > 70):
                risk_factors.append("Severe thunderstorm conditions")
                risk_score += 3
            
            # If significant risk detected
            if risk_score >= 3:
                event_type = "Severe Weather Event"
                if risk_score >= 6:
                    event_type = "Extreme Weather Event"
                
                predictions.append({
                    "timestamp": row['time'].isoformat(),
                    "event_type": event_type,
                    "risk_score": risk_score,
                    "risk_factors": risk_factors,
                    "conditions": {
                        "temperature": row['temperature_2m'],
                        "wind_speed": row['wind_speed_10m'],
                        "precipitation": row['precipitation'],
                        "humidity": row['relative_humidity_2m']
                    },
                    "confidence": min(risk_score * 15, 95)  # Convert to percentage
                })
        
        return predictions

class WatsonxAIIntegration:
    """Integrates with IBM watsonx.ai using official SDK"""
    
    def __init__(self, api_key: str, project_id: str, endpoint_url: str):
        self.api_key = api_key
        self.project_id = project_id  
        self.endpoint_url = endpoint_url
        self.client = None
        self.model = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize IBM watsonx.ai client and model"""
        try:
            print(f"DEBUG: Starting initialization...")
            print(f"DEBUG: API key: {self.api_key[:20] if self.api_key else 'None'}...")
            print(f"DEBUG: Project ID: {self.project_id}")
            print(f"DEBUG: Endpoint: {self.endpoint_url}")
            
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            
            # Create credentials
            credentials = Credentials(
                url=self.endpoint_url,
                api_key=self.api_key,
            )
            print("DEBUG: ✅ Credentials created successfully")
            
            # Create client
            self.client = APIClient(credentials)
            print("DEBUG: ✅ API client created successfully")
            
            # FIXED parameters for better responses
            params = {
                "decoding_method": "greedy",
                "max_new_tokens": 800,
                "temperature": 0.3,
                "top_p": 0.8,
                "repetition_penalty": 1.2,
            }
            
            # Initialize model
            print("DEBUG: Creating model inference...")
            self.model = ModelInference(
                model_id="ibm/granite-13b-instruct-v2",
                api_client=self.client,
                params=params,
                project_id=self.project_id,
                space_id=None,
                verify=False,
            )
            print("DEBUG: ✅ Model inference created successfully")
            
            # Test the model with a simple call
            print("DEBUG: Testing model with simple generation...")
            test_result = self.model.generate_text("Hello")
            print(f"DEBUG: ✅ Model test successful: '{test_result[:50]}...'")
            
            logger.info("✅ IBM watsonx.ai client initialized successfully")
            
        except Exception as e:
            print(f"DEBUG: ❌ Initialization failed at step: {e}")
            print(f"DEBUG: Exception type: {type(e).__name__}")
            import traceback
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            
            logger.error(f"❌ Failed to initialize IBM watsonx.ai client: {e}")
            self.client = None
            self.model = None
    
    def generate_weather_analysis(self, weather_data: Dict, predictions: List[Dict], location_name: str = None) -> str:
        """Use watsonx.ai to generate comprehensive weather analysis"""
        
        print(f"DEBUG: generate_weather_analysis called for {location_name}")
        print(f"DEBUG: self.model exists: {self.model is not None}")
        print(f"DEBUG: self.client exists: {self.client is not None}")
        
        # Better check - try to initialize if model is None
        if not self.model:
            print("DEBUG: Model not available, attempting to re-initialize...")
            self._initialize_client()
            
            if not self.model:
                print("DEBUG: Re-initialization failed, using fallback")
                return self._generate_fallback_analysis(weather_data, predictions, location_name)
        
        try:
            # Prepare context for RAG
            context = self._prepare_weather_context(weather_data, predictions)
            location = location_name if location_name else "the analyzed location"
            
            # Simple prompt
            prompt = f"""Analyze weather conditions for {location}.

    Data: {context}

    Write a clear weather report with:
    1. Current weather threats
    2. 7-day risk level (LOW/MODERATE/HIGH)  
    3. What residents should do to prepare
    4. Emergency actions if weather worsens
    5. Climate patterns for this location

    Keep the response under 500 words and focus on practical advice."""
            
            print(f"DEBUG: Calling model.generate_text...")
            
            result = self.model.generate_text(prompt)
            
            print(f"DEBUG: Model response received, length: {len(result) if result else 0}")
            
            # Better response cleaning
            if result and len(result.strip()) > 50:
                # Remove obvious gibberish
                if result.count('0') > len(result) * 0.3:
                    print("DEBUG: Detected gibberish, using fallback")
                    return self._generate_fallback_analysis(weather_data, predictions, location_name)
                
                # Clean the response
                cleaned = self._clean_ai_response_simple(result)
                
                if len(cleaned.strip()) > 50:
                    print("DEBUG: Returning cleaned AI response")
                    return cleaned
            
            print("DEBUG: AI response insufficient, using fallback")
            return self._generate_fallback_analysis(weather_data, predictions, location_name)
            
        except Exception as e:
            print(f"DEBUG: Exception during generation: {e}")
            logger.error(f"Error generating weather analysis: {e}")
            return self._generate_fallback_analysis(weather_data, predictions, location_name)

    def _get_location_context(self, location: str) -> str:
        """Get relevant context for location-specific analysis"""
        location_lower = location.lower()
        
        # Determine climate zone and typical hazards
        if any(term in location_lower for term in ['miami', 'florida', 'gulf coast', 'caribbean']):
            return "Subtropical coastal location. Hurricane season June-November. High humidity year-round."
        elif any(term in location_lower for term in ['tokyo', 'japan']):
            return "Temperate monsoon climate. Typhoon season June-October. Earthquakes possible."
        elif any(term in location_lower for term in ['sydney', 'australia']):
            return "Temperate oceanic climate. Summer storms Dec-Mar. Bushfire risk in hot, dry conditions."
        elif any(term in location_lower for term in ['paris', 'france', 'europe']):
            return "Temperate oceanic climate. Mild winters, warm summers. Occasional severe thunderstorms."
        elif any(term in location_lower for term in ['tornado alley', 'oklahoma', 'kansas', 'texas']):
            return "Continental climate. Peak tornado season March-June. Severe thunderstorms common."
        elif any(term in location_lower for term in ['california', 'nevada', 'arizona']):
            return "Arid to Mediterranean climate. Wildfire risk. Extreme heat in summer."
        elif any(term in location_lower for term in ['london', 'uk', 'britain', 'england']):
            return "Temperate maritime climate. Mild, wet winters. Occasional severe weather."
        elif any(term in location_lower for term in ['new york', 'northeast', 'boston']):
            return "Humid continental climate. Winter storms, summer thunderstorms. Hurricane risk."
        elif any(term in location_lower for term in ['canada', 'toronto', 'montreal']):
            return "Continental climate. Harsh winters, severe thunderstorms possible."
        else:
            return "Analyze based on the provided data and general meteorological principles."

    def _clean_ai_response_simple(self, response: str) -> str:
        """Simple response cleaning"""
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are obviously prompt echoes
            line_lower = line.lower()
            if any(skip in line_lower for skip in [
                'write a clear weather report', 'analyze weather conditions', 
                'keep the response under', 'focus on practical'
            ]):
                continue
                
            # Skip lines with too many repeated characters
            if len(set(line)) < 3:  # Less than 3 unique characters
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def generate_community_alert(self, prediction: Dict, location: str) -> str:
        """Generate community-specific weather alert"""
        if not self.model:
            return f"⚠️ WEATHER ALERT for {location}: {prediction['event_type']} predicted with risk level {prediction['risk_score']}/10. Please monitor weather updates and follow local emergency guidelines."
        
        # IMPROVED ALERT PROMPT
        prompt = f"""Create an emergency weather alert for {location}:

THREAT: {prediction['event_type']}
WHEN: {prediction['timestamp']}
RISK LEVEL: {prediction['risk_score']}/10 (High risk is 6+)
CONFIDENCE: {prediction['confidence']}%
FACTORS: {', '.join(prediction['risk_factors'])}

Write a clear, urgent alert that tells residents:
1. What weather threat is coming
2. When it will arrive
3. What actions to take immediately
4. Who is most at risk
5. Where to get updates

Use clear, direct language that non-experts can understand."""
        
        try:
            print(f"DEBUG: Generating alert for {location}...")
            result = self.model.generate_text(prompt)
            print(f"DEBUG: Alert response length: {len(result) if result else 0}")
            
            if result and len(result.strip()) > 20:
                return result.strip()
            else:
                return f"""🚨 WEATHER ALERT for {location}

THREAT: {prediction['event_type']} 
TIMING: {prediction['timestamp']}
RISK: {prediction['risk_score']}/10

TAKE ACTION: Monitor weather conditions closely. Prepare emergency supplies. Stay informed through local weather services.

MOST AT RISK: Outdoor workers, coastal residents, mobile home residents.

UPDATES: Follow local emergency management for latest information."""
                
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            return f"Weather Alert: {prediction['event_type']} expected for {location}"
    
    def get_historical_insights(self, extreme_events: List[Dict], location_name: str = None) -> str:
        """Generate insights from historical extreme weather patterns"""
        location = location_name if location_name else "the analyzed location"
        
        if not self.model:
            return f"Historical analysis for {location} requires AI capabilities. Basic patterns show seasonal weather variations."
            
        events_summary = json.dumps(extreme_events[:10], indent=2)
        
        prompt = f"""Analyze these historical extreme weather events for {location} to identify patterns and trends:
        
        {events_summary}
        
        Provide insights on:
        1. Seasonal patterns of extreme weather in {location}
        2. Frequency and intensity trends
        3. Types of events most common in {location}
        4. Climate change indicators for this region
        5. Recommendations for long-term resilience planning in {location}
        
        Base your analysis on meteorological science and climate research specific to this location."""
        
        try:
            result = self.model.generate_text(prompt)
            return result if result else f"Historical patterns analysis not available for {location}."
        except Exception as e:
            logger.error(f"Error generating historical insights: {e}")
            return f"Historical patterns analysis not available for {location}."
    
    def _prepare_weather_context(self, weather_data: Dict, predictions: List[Dict]) -> str:
        """Prepare weather data context for RAG"""
        context_parts = []
        
        if 'statistics' in weather_data:
            stats = weather_data['statistics']
            context_parts.append(f"""
            HISTORICAL STATISTICS:
            - Total extreme events detected: {stats.get('total_extreme_events', 0)}
            - Average wind speed: {stats.get('avg_wind_speed', 0):.1f} km/h
            - Maximum wind speed: {stats.get('max_wind_speed', 0):.1f} km/h
            - Total precipitation: {stats.get('total_precipitation', 0):.1f} mm
            - Temperature range: {stats.get('min_temperature', 0):.1f}°C to {stats.get('max_temperature', 0):.1f}°C
            """)
        
        if predictions:
            context_parts.append("UPCOMING PREDICTIONS:")
            for pred in predictions[:5]:  # Limit for token efficiency
                context_parts.append(f"""
                - {pred['timestamp']}: {pred['event_type']} (Risk: {pred['risk_score']}/10)
                  Factors: {', '.join(pred['risk_factors'])}
                """)
        
        return '\n'.join(context_parts)
    
    def _generate_fallback_analysis(self, weather_data: Dict, predictions: List[Dict], location_name: str = None) -> str:
        """Generate comprehensive fallback analysis"""
        stats = weather_data.get('statistics', {})
        location = location_name if location_name else "the analyzed location"
        
        # Determine risk levels
        max_wind = stats.get('max_wind_speed', 0)
        total_precip = stats.get('total_precipitation', 0)
        max_temp = stats.get('max_temperature', 0)
        min_temp = stats.get('min_temperature', 0)
        extreme_events = stats.get('total_extreme_events', 0)
        
        # Risk assessment
        wind_risk = "HIGH" if max_wind > 60 else "MODERATE" if max_wind > 30 else "LOW"
        temp_risk = "EXTREME" if (max_temp > 35 or min_temp < -10) else "MODERATE" if (max_temp > 30 or min_temp < 0) else "LOW"
        precip_risk = "HIGH" if total_precip > 200 else "MODERATE" if total_precip > 100 else "LOW"
        
        overall_risk = "HIGH" if any(risk == "HIGH" for risk in [wind_risk, temp_risk, precip_risk]) or extreme_events > 0 else "MODERATE" if any(risk == "MODERATE" for risk in [wind_risk, temp_risk, precip_risk]) else "LOW"
        
        # Location-specific guidance
        location_lower = location.lower()
        if any(term in location_lower for term in ['tokyo', 'japan']):
            seasonal_info = "Typhoon season June-October. Winter brings cold temperatures. Cherry blossom season in spring."
            hazards = "Typhoons, earthquakes, winter storms, summer heat waves"
        elif any(term in location_lower for term in ['sydney', 'australia']):
            seasonal_info = "Summer storm season December-March. Bushfire risk during hot, dry periods. Mild winters."
            hazards = "Bushfires, severe thunderstorms, hail, occasional flooding"
        elif any(term in location_lower for term in ['miami', 'florida']):
            seasonal_info = "Hurricane season June-November. Hot, humid summers. Mild winters."
            hazards = "Hurricanes, severe thunderstorms, flooding, extreme heat"
        elif any(term in location_lower for term in ['reykjavik', 'iceland']):
            seasonal_info = "Harsh winters with snow and ice. Cool summers. Northern lights season October-March."
            hazards = "Winter storms, ice, strong winds, volcanic ash (rare)"
        elif any(term in location_lower for term in ['paris', 'france']):
            seasonal_info = "Mild winters, warm summers. Occasional severe thunderstorms in summer."
            hazards = "Winter frost, summer heat waves, thunderstorms, occasional flooding"
        elif any(term in location_lower for term in ['london', 'uk']):
            seasonal_info = "Mild, wet climate year-round. Occasional winter snow. Summer can bring heat waves."
            hazards = "Winter storms, occasional snow, summer heat waves, flooding"
        else:
            seasonal_info = "Monitor local weather patterns and seasonal changes."
            hazards = "Various weather hazards depending on season and location"
        
        analysis = f"""**COMPREHENSIVE WEATHER ANALYSIS - {location.upper()}**

    **🌊 CURRENT THREATS:** 
    {location} shows {extreme_events} extreme weather events in recent data. Current conditions indicate {overall_risk} overall risk based on observed weather patterns.

    **📈 7-DAY RISK ASSESSMENT: {overall_risk}**
    - Wind Risk: {wind_risk} (maximum observed: {max_wind:.1f} km/h)
    - Temperature Risk: {temp_risk} (range: {min_temp:.1f}°C to {max_temp:.1f}°C)
    - Precipitation Risk: {precip_risk} (total: {total_precip:.1f}mm)

    **🎯 COMMUNITY PREPAREDNESS:**
    Residents should maintain emergency supplies:
    - 72-hour supply of water (4 liters per person per day)
    - Non-perishable food, battery-powered radio, flashlights
    - First aid kit, medications, important documents in waterproof container
    - {seasonal_info}

    **🚨 EMERGENCY PROTOCOLS:**
    - Monitor local weather alerts and emergency management updates
    - If severe weather threatens: move to interior rooms away from windows
    - Avoid flooded areas and downed power lines
    - Have evacuation routes planned and emergency contacts ready
    - Keep mobile devices charged and have backup power sources

    **📊 CLIMATE ANALYSIS:**
    Recent weather patterns for {location}:
    - Wind conditions average {stats.get('avg_wind_speed', 0):.1f} km/h (indicating {"stable" if max_wind < 40 else "variable"} atmospheric conditions)
    - Temperature variation shows {"normal" if abs(max_temp - min_temp) < 25 else "significant"} seasonal range
    - Precipitation levels are {"above normal" if total_precip > 150 else "below normal" if total_precip < 50 else "typical"} for the analysis period

    **⚠️ LOCAL HAZARDS:** {hazards}

    **RECOMMENDATIONS:** {seasonal_info} Stay informed through official weather services and local emergency management."""

        return analysis

class ExtremeWeatherRAGSystem:
    """Main RAG system for extreme weather prediction and community alerts"""
    
    def __init__(self, watsonx_api_key: str, watsonx_project_id: str, 
                 watsonx_endpoint: str = "https://us-south.ml.cloud.ibm.com"):
        self.weather_collector = WeatherDataCollector()
        self.detector = ExtremeWeatherDetector()
        self.ai_integration = WatsonxAIIntegration(
            api_key=watsonx_api_key,
            project_id=watsonx_project_id,
            endpoint_url=watsonx_endpoint
        )
        
    def analyze_location(self, lat: float, lon: float, location_name: str, 
                    days_history: int = 30) -> Dict:
        """
        Complete analysis of extreme weather risks for a location
        """
        logger.info(f"Starting analysis for {location_name} ({lat}, {lon})")
        
        # Calculate date range for historical data - PROPERLY FIXED
        from datetime import datetime, timedelta
        
        today = datetime.now().date()
        # Use end of 2023 as end date (guaranteed to have data)
        end_date = datetime(2023, 12, 31).date()
        start_date = end_date - timedelta(days=days_history)
        
        logger.info(f"Requesting historical data from {start_date} to {end_date}")
        
        # Collect historical data
        logger.info("Collecting historical weather data...")
        historical_data = self.weather_collector.get_historical_data(
            lat, lon, start_date.isoformat(), end_date.isoformat()
        )
        
        # Collect forecast data
        logger.info("Collecting forecast data...")
        forecast_data = self.weather_collector.get_current_forecast(lat, lon, days=7)
        
        # Analyze patterns
        logger.info("Analyzing historical extreme weather patterns...")
        historical_analysis = self.detector.analyze_historical_patterns(historical_data)
        
        # Generate predictions
        logger.info("Generating extreme weather predictions...")
        predictions = self.detector.predict_extreme_events(forecast_data, historical_analysis)
        
        # Generate AI-powered analysis (PASS LOCATION NAME)
        logger.info("Generating AI analysis with watsonx.ai...")
        ai_analysis = self.ai_integration.generate_weather_analysis(
            historical_analysis, predictions, location_name  # ADD location_name here
        )
        
        # Generate community alerts for high-risk predictions
        alerts = []
        for prediction in predictions:
            if prediction['risk_score'] >= 5:  # High risk threshold
                alert = self.ai_integration.generate_community_alert(prediction, location_name)
                alerts.append({
                    "timestamp": prediction['timestamp'],
                    "alert_text": alert,
                    "risk_score": prediction['risk_score']
                })
        
        # Historical insights (PASS LOCATION NAME)
        if historical_analysis.get('extreme_events'):
            historical_insights = self.ai_integration.get_historical_insights(
                historical_analysis['extreme_events'], location_name  # ADD location_name here
            )
        else:
            historical_insights = f"No significant extreme weather events detected in historical data for {location_name}."
        
        return {
            "location": {
                "name": location_name,
                "coordinates": {"lat": lat, "lon": lon}
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "historical_analysis": historical_analysis,
            "predictions": predictions,
            "ai_analysis": ai_analysis,
            "community_alerts": alerts,
            "historical_insights": historical_insights,
            "data_sources": {
                "weather_data": "Open-Meteo API",
                "ai_analysis": "IBM watsonx.ai",
                "analysis_period": f"{start_date} to {end_date}"
            }
        }
    
    def monitor_location(self, lat: float, lon: float, location_name: str, 
                        check_interval: int = 3600) -> None:
        """
        Continuously monitor a location for extreme weather threats
        
        Args:
            lat: Latitude  
            lon: Longitude
            location_name: Name of location
            check_interval: Check interval in seconds (default 1 hour)
        """
        logger.info(f"Starting continuous monitoring for {location_name}")
        
        while True:
            try:
                # Get latest forecast
                forecast_data = self.weather_collector.get_current_forecast(lat, lon, days=3)
                
                # Quick threat assessment
                predictions = self.detector.predict_extreme_events(forecast_data, {})
                
                # Check for immediate threats (next 6 hours)
                immediate_threats = [
                    p for p in predictions 
                    if pd.to_datetime(p['timestamp']) <= datetime.now() + timedelta(hours=6)
                    and p['risk_score'] >= 6
                ]
                
                if immediate_threats:
                    logger.warning(f"IMMEDIATE THREAT DETECTED for {location_name}")
                    for threat in immediate_threats:
                        alert = self.ai_integration.generate_community_alert(threat, location_name)
                        logger.warning(f"ALERT: {alert}")
                        
                        # In a real system, this would trigger notifications, SMS, etc.
                        print(f"\n🚨 EXTREME WEATHER ALERT for {location_name} 🚨")
                        print(f"Time: {threat['timestamp']}")
                        print(f"Risk Score: {threat['risk_score']}/10")
                        print(f"Alert: {alert}\n")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    """Example usage of the Extreme Weather RAG System"""
    
    # Configuration - Replace with your watsonx.ai credentials
    WATSONX_API_KEY = "your_watsonx_api_key_here"
    WATSONX_PROJECT_ID = "your_project_id_here"
    WATSONX_ENDPOINT = "https://us-south.ml.cloud.ibm.com"
    
    # Test locations prone to extreme weather
    test_locations = [
        {"name": "Miami, FL", "lat": 25.7617, "lon": -80.1918},  # Hurricane prone
        {"name": "Moore, OK", "lat": 35.3395, "lon": -97.4864},  # Tornado Alley
        {"name": "Phoenix, AZ", "lat": 33.4484, "lon": -112.0740},  # Heat waves
        {"name": "Buffalo, NY", "lat": 42.8864, "lon": -78.8784},  # Lake effect snow/blizzards
    ]
    
    # Initialize the RAG system
    try:
        rag_system = ExtremeWeatherRAGSystem(
            watsonx_api_key=WATSONX_API_KEY,
            watsonx_project_id=WATSONX_PROJECT_ID,
            watsonx_endpoint=WATSONX_ENDPOINT
        )
        
        print("🌪️ Extreme Weather Prediction RAG System")
        print("=" * 50)
        
        # Analyze each test location
        for location in test_locations:
            print(f"\n📍 Analyzing {location['name']}...")
            
            analysis = rag_system.analyze_location(
                lat=location['lat'],
                lon=location['lon'], 
                location_name=location['name'],
                days_history=30
            )
            
            print(f"\n📊 Analysis Results for {location['name']}:")
            print(f"Historical extreme events: {analysis['historical_analysis'].get('statistics', {}).get('total_extreme_events', 0)}")
            print(f"Predictions for next 7 days: {len(analysis['predictions'])}")
            print(f"High-risk alerts: {len(analysis['community_alerts'])}")
            
            # Show AI analysis snippet
            ai_analysis = analysis['ai_analysis']
            print(f"\n🤖 AI Analysis Preview:")
            print(ai_analysis[:300] + "..." if len(ai_analysis) > 300 else ai_analysis)
            
            # Show any immediate alerts
            if analysis['community_alerts']:
                print(f"\n🚨 ACTIVE ALERTS:")
                for alert in analysis['community_alerts']:
                    print(f"- Risk Level {alert['risk_score']}/10 at {alert['timestamp']}")
        
        # Optionally start monitoring (uncomment to enable)
        # print("\n🔄 Starting continuous monitoring for Miami...")
        # rag_system.monitor_location(25.7617, -80.1918, "Miami, FL", check_interval=1800)
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        print("\n❌ Error: Please check your watsonx.ai credentials and try again.")
        print("Make sure to replace the placeholder API key and project ID with your actual credentials.")

if __name__ == "__main__":
    main()

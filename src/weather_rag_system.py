#!/usr/bin/env python3
"""
Extreme Weather Event Prediction RAG System
Clean and optimized version for IBM watsonx.ai Hackathon
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Collects weather data from Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.historical_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = f"{self.base_url}/forecast"
        self.session = requests.Session()
        
        # Configure retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def get_historical_data(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict:
        """Retrieve historical weather data"""
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max",
            "timezone": "auto"
        }
        
        try:
            response = self.session.get(self.historical_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Successfully retrieved historical data")
                return data
            else:
                logger.error(f"API returned status {response.status_code}")
                return self._create_mock_historical_data(lat, lon)
                
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching historical data, using mock data")
            return self._create_mock_historical_data(lat, lon)
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return self._create_mock_historical_data(lat, lon)
    
    def get_current_forecast(self, lat: float, lon: float, days: int = 7) -> Dict:
        """Get current weather forecast"""
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max",
            "forecast_days": min(days, 16),
            "timezone": "auto"
        }
        
        try:
            response = self.session.get(self.forecast_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Successfully retrieved forecast data")
                return data
            else:
                logger.error(f"Forecast API returned status {response.status_code}")
                return self._create_mock_forecast_data(lat, lon, days)
                
        except Exception as e:
            logger.error(f"Error fetching forecast data: {e}")
            return self._create_mock_forecast_data(lat, lon, days)
    
    def _create_mock_historical_data(self, lat: float, lon: float) -> Dict:
        """Create realistic mock historical data when API fails"""
        
        dates = [(datetime.now() - timedelta(days=i)).isoformat() for i in range(30, 0, -1)]
        base_temp = 20 - abs(lat) * 0.5
        
        hourly_data = {
            "time": [f"{date}T{hour:02d}:00" for date in dates[:30] for hour in range(24)],
            "temperature_2m": [],
            "relative_humidity_2m": [],
            "precipitation": [],
            "wind_speed_10m": [],
            "wind_gusts_10m": []
        }
        
        for i in range(len(hourly_data["time"])):
            temp_var = np.random.normal(0, 3)
            hourly_data["temperature_2m"].append(base_temp + temp_var)
            hourly_data["relative_humidity_2m"].append(max(20, min(95, 60 + np.random.normal(0, 15))))
            
            if np.random.random() < 0.1:
                hourly_data["precipitation"].append(np.random.exponential(2))
            else:
                hourly_data["precipitation"].append(0)
            
            hourly_data["wind_speed_10m"].append(max(0, np.random.normal(10, 5)))
            hourly_data["wind_gusts_10m"].append(hourly_data["wind_speed_10m"][-1] * (1 + np.random.random() * 0.5))
        
        return {
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_data,
            "daily": {
                "time": dates[:30],
                "temperature_2m_max": [base_temp + 5 + np.random.normal(0, 2) for _ in range(30)],
                "temperature_2m_min": [base_temp - 5 + np.random.normal(0, 2) for _ in range(30)],
                "precipitation_sum": [np.random.exponential(1) if np.random.random() < 0.3 else 0 for _ in range(30)]
            }
        }
    
    def _create_mock_forecast_data(self, lat: float, lon: float, days: int) -> Dict:
        """Create realistic mock forecast data when API fails"""
        
        dates = [(datetime.now() + timedelta(days=i)).isoformat()[:10] for i in range(days)]
        base_temp = 20 - abs(lat) * 0.5
        
        hourly_data = {
            "time": [f"{date}T{hour:02d}:00" for date in dates for hour in range(24)],
            "temperature_2m": [],
            "relative_humidity_2m": [],
            "precipitation": [],
            "wind_speed_10m": [],
            "wind_gusts_10m": []
        }
        
        for i in range(len(hourly_data["time"])):
            temp_var = np.random.normal(0, 2)
            hourly_data["temperature_2m"].append(base_temp + temp_var)
            hourly_data["relative_humidity_2m"].append(max(30, min(90, 65 + np.random.normal(0, 10))))
            
            if np.random.random() < 0.05:
                hourly_data["precipitation"].append(np.random.exponential(1))
            else:
                hourly_data["precipitation"].append(0)
            
            hourly_data["wind_speed_10m"].append(max(0, np.random.normal(8, 3)))
            hourly_data["wind_gusts_10m"].append(hourly_data["wind_speed_10m"][-1] * (1 + np.random.random() * 0.3))
        
        return {
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_data,
            "daily": {
                "time": dates,
                "temperature_2m_max": [base_temp + 4 + np.random.normal(0, 1.5) for _ in range(days)],
                "temperature_2m_min": [base_temp - 4 + np.random.normal(0, 1.5) for _ in range(days)],
                "precipitation_sum": [np.random.exponential(0.5) if np.random.random() < 0.2 else 0 for _ in range(days)]
            }
        }

class ExtremeWeatherDetector:
    """Analyzes weather data to detect extreme weather patterns"""
    
    def __init__(self):
        self.thresholds = {
            "hurricane_wind_speed": 119,
            "severe_thunderstorm_wind": 93,
            "heavy_rain_daily": 50,
            "extreme_rain_hourly": 25,
            "heat_wave_temp": 35,
            "cold_wave_temp": -18,
            "blizzard_wind": 56,
        }
    
    def analyze_historical_patterns(self, weather_data: Dict) -> Dict:
        """Analyze historical data to identify extreme weather patterns"""
        
        if not weather_data or 'hourly' not in weather_data:
            return self._create_default_analysis()
        
        try:
            hourly = weather_data['hourly']
            df_hourly = pd.DataFrame(hourly)
            if 'time' in df_hourly.columns:
                df_hourly['time'] = pd.to_datetime(df_hourly['time'])
            
            extreme_events = []
            
            # Detect extreme wind events
            if 'wind_speed_10m' in df_hourly.columns:
                wind_speeds = pd.to_numeric(df_hourly['wind_speed_10m'], errors='coerce').fillna(0)
                high_winds = df_hourly[wind_speeds > self.thresholds['severe_thunderstorm_wind']]
                
                for _, event in high_winds.iterrows():
                    wind_speed = event.get('wind_speed_10m', 0)
                    if pd.notna(wind_speed):
                        extreme_events.append({
                            "type": "High Wind Event",
                            "timestamp": event.get('time', datetime.now()).isoformat(),
                            "wind_speed": float(wind_speed),
                            "severity": "Hurricane" if wind_speed > self.thresholds['hurricane_wind_speed'] else "Severe"
                        })
            
            # Detect precipitation events
            if 'precipitation' in df_hourly.columns:
                precipitation = pd.to_numeric(df_hourly['precipitation'], errors='coerce').fillna(0)
                heavy_rain = df_hourly[precipitation > self.thresholds['extreme_rain_hourly']]
                
                for _, event in heavy_rain.iterrows():
                    precip_amount = event.get('precipitation', 0)
                    if pd.notna(precip_amount):
                        extreme_events.append({
                            "type": "Heavy Precipitation",
                            "timestamp": event.get('time', datetime.now()).isoformat(),
                            "precipitation": float(precip_amount),
                            "severity": "Extreme" if precip_amount > 50 else "Heavy"
                        })
            
            # Detect temperature extremes
            if 'temperature_2m' in df_hourly.columns:
                temperatures = pd.to_numeric(df_hourly['temperature_2m'], errors='coerce').fillna(20)
                heat_events = df_hourly[temperatures > self.thresholds['heat_wave_temp']]
                cold_events = df_hourly[temperatures < self.thresholds['cold_wave_temp']]
                
                for _, event in heat_events.iterrows():
                    extreme_events.append({
                        "type": "Heat Wave",
                        "timestamp": event.get('time', datetime.now()).isoformat(),
                        "temperature": float(event.get('temperature_2m', 0)),
                        "severity": "Extreme" if event.get('temperature_2m', 0) > 40 else "High"
                    })
                
                for _, event in cold_events.iterrows():
                    extreme_events.append({
                        "type": "Cold Wave", 
                        "timestamp": event.get('time', datetime.now()).isoformat(),
                        "temperature": float(event.get('temperature_2m', 0)),
                        "severity": "Extreme" if event.get('temperature_2m', 0) < -25 else "Severe"
                    })
            
            # Calculate statistics
            stats = self._calculate_statistics(df_hourly)
            stats["total_extreme_events"] = len(extreme_events)
            
            return {
                "extreme_events": extreme_events,
                "statistics": stats,
                "analysis_period": {
                    "start": df_hourly['time'].min().isoformat() if 'time' in df_hourly.columns and not df_hourly.empty else (datetime.now() - timedelta(days=30)).isoformat(),
                    "end": df_hourly['time'].max().isoformat() if 'time' in df_hourly.columns and not df_hourly.empty else datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return self._create_default_analysis()
    
    def _calculate_statistics(self, df_hourly: pd.DataFrame) -> Dict:
        """Calculate weather statistics from dataframe"""
        
        stats = {}
        
        if 'wind_speed_10m' in df_hourly.columns:
            wind_speeds = pd.to_numeric(df_hourly['wind_speed_10m'], errors='coerce').fillna(0)
            stats.update({
                "avg_wind_speed": float(wind_speeds.mean()),
                "max_wind_speed": float(wind_speeds.max())
            })
        else:
            stats.update({"avg_wind_speed": 10.0, "max_wind_speed": 25.0})
        
        if 'precipitation' in df_hourly.columns:
            precipitation = pd.to_numeric(df_hourly['precipitation'], errors='coerce').fillna(0)
            stats["total_precipitation"] = float(precipitation.sum())
        else:
            stats["total_precipitation"] = 30.0
        
        if 'temperature_2m' in df_hourly.columns:
            temperatures = pd.to_numeric(df_hourly['temperature_2m'], errors='coerce').fillna(20)
            stats.update({
                "avg_temperature": float(temperatures.mean()),
                "max_temperature": float(temperatures.max()),
                "min_temperature": float(temperatures.min())
            })
        else:
            stats.update({"avg_temperature": 20.0, "max_temperature": 28.0, "min_temperature": 12.0})
        
        return stats
    
    def _create_default_analysis(self) -> Dict:
        """Create default analysis when data is unavailable"""
        return {
            "extreme_events": [],
            "statistics": {
                "total_extreme_events": 0,
                "avg_wind_speed": 10.0,
                "max_wind_speed": 25.0,
                "total_precipitation": 30.0,
                "avg_temperature": 20.0,
                "max_temperature": 28.0,
                "min_temperature": 12.0
            },
            "analysis_period": {
                "start": (datetime.now() - timedelta(days=30)).isoformat(),
                "end": datetime.now().isoformat()
            }
        }
    
    def predict_extreme_events(self, forecast_data: Dict, historical_analysis: Dict) -> List[Dict]:
        """Predict potential extreme weather events based on forecast"""
        
        if not forecast_data or 'hourly' not in forecast_data:
            return []
        
        try:
            predictions = []
            hourly = forecast_data['hourly']
            df_forecast = pd.DataFrame(hourly)
            
            if 'time' in df_forecast.columns:
                df_forecast['time'] = pd.to_datetime(df_forecast['time'])
            
            for i, row in df_forecast.iterrows():
                risk_factors = []
                risk_score = 0
                
                wind_speed = pd.to_numeric(row.get('wind_speed_10m', 0), errors='coerce')
                if pd.isna(wind_speed):
                    wind_speed = 0
                
                temperature = pd.to_numeric(row.get('temperature_2m', 20), errors='coerce')
                if pd.isna(temperature):
                    temperature = 20
                
                precipitation = pd.to_numeric(row.get('precipitation', 0), errors='coerce')
                if pd.isna(precipitation):
                    precipitation = 0
                
                humidity = pd.to_numeric(row.get('relative_humidity_2m', 60), errors='coerce')
                if pd.isna(humidity):
                    humidity = 60
                
                # Risk assessment
                if wind_speed > self.thresholds['severe_thunderstorm_wind']:
                    risk_factors.append("High wind speeds predicted")
                    risk_score += 3
                    
                if wind_speed > self.thresholds['hurricane_wind_speed']:
                    risk_factors.append("Hurricane-force winds")
                    risk_score += 4
                
                if precipitation > self.thresholds['extreme_rain_hourly']:
                    risk_factors.append("Heavy precipitation expected")
                    risk_score += 3
                
                if temperature > self.thresholds['heat_wave_temp']:
                    risk_factors.append("Extreme heat conditions")
                    risk_score += 2
                elif temperature < self.thresholds['cold_wave_temp']:
                    risk_factors.append("Extreme cold conditions")
                    risk_score += 2
                
                if precipitation > 5 and wind_speed > 50 and humidity > 70:
                    risk_factors.append("Severe thunderstorm conditions")
                    risk_score += 3
                
                if risk_score >= 3:
                    event_type = "Severe Weather Event"
                    if risk_score >= 6:
                        event_type = "Extreme Weather Event"
                    
                    timestamp = row.get('time', datetime.now())
                    if pd.isna(timestamp):
                        timestamp = datetime.now()
                    
                    predictions.append({
                        "timestamp": timestamp.isoformat(),
                        "event_type": event_type,
                        "risk_score": min(risk_score, 10),
                        "risk_factors": risk_factors,
                        "conditions": {
                            "temperature": float(temperature),
                            "wind_speed": float(wind_speed),
                            "precipitation": float(precipitation),
                            "humidity": float(humidity)
                        },
                        "confidence": min(risk_score * 15, 95)
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting extreme events: {e}")
            return []

class WatsonxAIIntegration:
    """Integrates with IBM watsonx.ai using official SDK"""
    
    def __init__(self, api_key: str, project_id: str, endpoint_url: str):
        self.api_key = api_key
        self.project_id = project_id  
        self.endpoint_url = endpoint_url
        self.client = None
        self.model = None
        self.is_initialized = False
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize IBM watsonx.ai client"""
        try:
            if (self.api_key == 'your_api_key_here' or 
                self.project_id == 'your_project_id_here' or
                not self.api_key or not self.project_id):
                logger.warning("watsonx.ai credentials not configured properly")
                self.is_initialized = False
                return False
            
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            
            credentials = Credentials(
                url=self.endpoint_url,
                api_key=self.api_key,
            )
            
            self.client = APIClient(credentials)
            
            params = {
                "decoding_method": "greedy",
                "max_new_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
            }
            
            self.model = ModelInference(
                model_id="ibm/granite-13b-instruct-v2",
                api_client=self.client,
                params=params,
                project_id=self.project_id,
                space_id=None,
                verify=False,
            )
            
            test_result = self.model.generate_text("Test")
            if test_result:
                logger.info("IBM watsonx.ai client initialized successfully")
                self.is_initialized = True
                return True
            else:
                logger.error("watsonx.ai test failed")
                self.is_initialized = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize watsonx.ai client: {e}")
            self.is_initialized = False
            return False
    
    def generate_weather_analysis(self, weather_data: Dict, predictions: List[Dict], location_name: str = None) -> str:
        """Generate weather analysis with AI or fallback"""
        
        if not self.is_initialized or not self.model:
            return self._generate_fallback_analysis(weather_data, predictions, location_name)
        
        try:
            context = self._prepare_weather_context(weather_data, predictions)
            location = location_name if location_name else "the analyzed location"
            
            prompt = f"""Weather analysis for {location}:

{context}

Provide a brief weather report including:
1. Main weather threats
2. Risk level (LOW/MODERATE/HIGH)
3. Safety recommendations
4. Climate patterns for this location

Keep response under 300 words."""
            
            result = self.model.generate_text(prompt)
            
            if result and len(result.strip()) > 50:
                return self._clean_ai_response(result)
            else:
                return self._generate_fallback_analysis(weather_data, predictions, location_name)
                
        except Exception as e:
            logger.error(f"Error generating weather analysis: {e}")
            return self._generate_fallback_analysis(weather_data, predictions, location_name)

    def _generate_fallback_analysis(self, weather_data: Dict, predictions: List[Dict], location_name: str) -> str:
        """Generate fallback analysis when AI is unavailable"""
        
        location = location_name if location_name else "this location"
        stats = weather_data.get('statistics', {})
        
        extreme_events = stats.get('total_extreme_events', 0)
        max_wind = stats.get('max_wind_speed', 0)
        max_temp = stats.get('max_temperature', 20)
        min_temp = stats.get('min_temperature', 10)
        
        risk_level = "LOW"
        if extreme_events > 0 or max_wind > 60 or max_temp > 35 or min_temp < -10:
            risk_level = "HIGH"
        elif max_wind > 30 or max_temp > 30 or min_temp < 0:
            risk_level = "MODERATE"
        
        high_risk_predictions = [p for p in predictions if p.get('risk_score', 0) >= 5]
        
        analysis = f"""WEATHER ANALYSIS - {location.upper()}

CURRENT THREATS: 
Analysis shows {extreme_events} extreme events detected in recent historical data. Current atmospheric conditions indicate {risk_level} overall risk.

7-DAY RISK ASSESSMENT: {risk_level}
Wind conditions: Maximum speeds of {max_wind:.1f} km/h recorded
Temperature extremes: Range from {min_temp:.1f}°C to {max_temp:.1f}°C  
Weather events: {len(high_risk_predictions)} high-risk events predicted

SAFETY RECOMMENDATIONS:
{"URGENT: Take shelter immediately if severe weather approaches. Avoid flooded areas and downed power lines." if risk_level == "HIGH" else "Stay informed about changing weather conditions. Monitor local weather alerts."}

CLIMATE CONTEXT:
Recent weather patterns show {"significant atmospheric instability" if extreme_events > 0 else "generally stable conditions"} for {location}."""

        return analysis

    def _prepare_weather_context(self, weather_data: Dict, predictions: List[Dict]) -> str:
        """Prepare weather data context"""
        context_parts = []
        
        if 'statistics' in weather_data:
            stats = weather_data['statistics']
            context_parts.append(f"Historical: {stats.get('total_extreme_events', 0)} extreme events, max wind {stats.get('max_wind_speed', 0):.1f} km/h")
        
        if predictions:
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            context_parts.append(f"Forecast: {len(high_risk)} high-risk events predicted")
        
        return '; '.join(context_parts)

    def _clean_ai_response(self, response: str) -> str:
        """Clean AI response"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return '\n'.join(lines)

class ExtremeWeatherRAGSystem:
    """Main RAG system for extreme weather prediction"""
    
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
        """Complete analysis of extreme weather risks for a location"""
        
        logger.info(f"Starting analysis for {location_name} ({lat}, {lon})")
        
        # Get historical data
        today = datetime.now().date()
        end_date = datetime(2023, 12, 31).date()
        start_date = end_date - timedelta(days=days_history)
        
        historical_data = self.weather_collector.get_historical_data(
            lat, lon, start_date.isoformat(), end_date.isoformat()
        )
        
        # Get forecast data
        forecast_data = self.weather_collector.get_current_forecast(lat, lon, days=7)
        
        # Analyze patterns
        historical_analysis = self.detector.analyze_historical_patterns(historical_data)
        predictions = self.detector.predict_extreme_events(forecast_data, historical_analysis)
        
        # Generate AI analysis
        ai_analysis = self.ai_integration.generate_weather_analysis(
            historical_analysis, predictions, location_name
        )
        
        # Generate alerts for high-risk events
        alerts = []
        for prediction in predictions:
            if prediction['risk_score'] >= 5:
                alerts.append({
                    "timestamp": prediction['timestamp'],
                    "risk_score": prediction['risk_score'],
                    "event_type": prediction['event_type']
                })
        
        return {
            "location": {
                "name": location_name,
                "coordinates": {"lat": lat, "lon": lon}
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "historical_analysis": historical_analysis,
            "predictions": predictions,
            "ai_analysis": ai_analysis,
            "community_alerts": alerts
        }
    
    def quick_analysis(self, lat: float, lon: float, location_name: str) -> str:
        """Quick analysis for faster responses"""
        
        try:
            forecast_data = self.weather_collector.get_current_forecast(lat, lon, days=3)
            
            if not forecast_data:
                return f"Unable to retrieve weather data for {location_name}."
            
            predictions = self.detector.predict_extreme_events(forecast_data, {})
            
            if not predictions:
                return f"No immediate weather threats detected for {location_name} in the next 3 days."
            
            high_risk = [p for p in predictions if p['risk_score'] >= 5]
            
            if high_risk:
                response = f"Weather Alert - {location_name}\n\n"
                for pred in high_risk[:2]:
                    response += f"• {pred['event_type']} - Risk {pred['risk_score']}/10\n"
                    response += f"  Time: {pred['timestamp']}\n"
                    response += f"  Factors: {', '.join(pred['risk_factors'][:2])}\n\n"
            else:
                response = f"Low risk weather conditions predicted for {location_name}"
            
            return response
            
        except Exception as e:
            logger.error(f"Quick analysis error: {e}")
            return f"Quick analysis failed for {location_name}: {str(e)}"
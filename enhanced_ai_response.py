#!/usr/bin/env python3
"""
Enhanced AI Response Generator for Weather RAG System
Focused on current conditions, forecast, and intelligent predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedWeatherAnalyzer:
    """Enhanced weather analysis with focused predictions"""
    
    def __init__(self):
        self.severity_thresholds = {
            'hurricane': {'wind': 119, 'pressure_drop': 20},
            'severe_storm': {'wind': 93, 'precipitation': 25},
            'heat_wave': {'temperature': 35, 'duration': 3},
            'cold_snap': {'temperature': -18, 'duration': 2},
            'heavy_rain': {'precipitation': 50, 'rate': 25},
            'drought': {'precipitation': 5, 'duration': 14}
        }
    
    def generate_current_conditions(self, forecast_data: Dict, location_name: str) -> str:
        """Generate current weather conditions from live data"""
        
        if not forecast_data or 'hourly' not in forecast_data:
            return "Current weather data unavailable"
        
        try:
            hourly_data = forecast_data['hourly']
            df_hourly = pd.DataFrame(hourly_data)
            
            if 'time' in df_hourly.columns:
                df_hourly['time'] = pd.to_datetime(df_hourly['time'])
                current_hour = df_hourly.iloc[0]
            else:
                current_hour = df_hourly.iloc[0]
            
            temp = self._safe_float(current_hour.get('temperature_2m'), 20)
            humidity = self._safe_float(current_hour.get('relative_humidity_2m'), 60)
            wind_speed = self._safe_float(current_hour.get('wind_speed_10m'), 10)
            precipitation = self._safe_float(current_hour.get('precipitation'), 0)
            pressure = self._safe_float(current_hour.get('surface_pressure'), 1013)
            
            conditions = f"CURRENT CONDITIONS - {location_name.upper()}\n"
            conditions += f"Temperature: {temp:.1f}°C\n"
            conditions += f"Humidity: {humidity:.0f}%\n"
            conditions += f"Wind Speed: {wind_speed:.1f} km/h\n"
            conditions += f"Precipitation: {precipitation:.1f} mm/h\n"
            conditions += f"Pressure: {pressure:.0f} hPa"
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error generating current conditions: {e}")
            return "Current weather conditions unavailable"
    
    def generate_7day_forecast_summary(self, forecast_data: Dict, location_name: str) -> str:
        """Generate concise 7-day forecast summary"""
        
        if not forecast_data or 'daily' not in forecast_data:
            return "7-day forecast unavailable"
        
        try:
            daily_data = forecast_data['daily']
            df_daily = pd.DataFrame(daily_data)
            
            if 'time' in df_daily.columns:
                df_daily['time'] = pd.to_datetime(df_daily['time'])
            
            forecast = "7-DAY FORECAST SUMMARY\n"
            
            for i, row in df_daily.head(7).iterrows():
                date = row.get('time', datetime.now() + timedelta(days=i))
                if pd.isna(date):
                    date = datetime.now() + timedelta(days=i)
                
                day_name = date.strftime('%a')
                temp_max = self._safe_float(row.get('temperature_2m_max'), 25)
                temp_min = self._safe_float(row.get('temperature_2m_min'), 15)
                precip = self._safe_float(row.get('precipitation_sum'), 0)
                wind_max = self._safe_float(row.get('wind_speed_10m_max'), 10)
                
                precip_status = "Rain" if precip > 1 else "Dry"
                wind_status = "Windy" if wind_max > 25 else "Calm"
                
                forecast += f"{day_name}: {temp_min:.0f}-{temp_max:.0f}°C, {precip_status}, {wind_status}\n"
            
            return forecast.strip()
            
        except Exception as e:
            logger.error(f"Error generating forecast summary: {e}")
            return "7-day forecast summary unavailable"
    
    def predict_extreme_weather_events(self, forecast_data: Dict, historical_data: Dict, location_name: str) -> str:
        """Predict extreme weather events using historical and live data"""
        
        try:
            predictions = "EXTREME WEATHER PREDICTIONS\n"
            events_found = False
            
            # Short-term predictions (next 7 days)
            short_term = self._analyze_short_term_threats(forecast_data)
            if short_term:
                predictions += "Next 7 Days:\n"
                for event in short_term[:2]:
                    predictions += f"- {event['type']}: {event['timing']} (Confidence: {event['confidence']}%)\n"
                events_found = True
            
            # Long-term predictions based on historical patterns
            long_term = self._analyze_seasonal_patterns(historical_data, location_name)
            if long_term:
                predictions += "Seasonal Outlook:\n"
                for event in long_term[:2]:
                    predictions += f"- {event['type']}: {event['timeframe']} (Historical probability: {event['probability']}%)\n"
                events_found = True
            
            if not events_found:
                predictions += "No significant extreme weather events predicted in available forecast period"
            
            return predictions.strip()
            
        except Exception as e:
            logger.error(f"Error predicting extreme events: {e}")
            return "Extreme weather prediction analysis unavailable"
    
    def generate_weather_intelligence_summary(self, forecast_data: Dict, historical_data: Dict, location_name: str) -> str:
        """Generate general weather intelligence summary"""
        
        try:
            stats = historical_data.get('statistics', {})
            extreme_events = stats.get('total_extreme_events', 0)
            
            summary = f"WEATHER INTELLIGENCE - {location_name.upper()}\n"
            
            # Risk assessment
            if extreme_events > 5:
                risk_level = "HIGH"
            elif extreme_events > 2:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            summary += f"Historical Risk Level: {risk_level}\n"
            summary += f"Extreme Events (Recent): {extreme_events}\n"
            
            # Climate characteristics
            avg_temp = stats.get('avg_temperature', 20)
            max_wind = stats.get('max_wind_speed', 25)
            
            if avg_temp > 30:
                climate = "Hot climate zone"
            elif avg_temp < 10:
                climate = "Cold climate zone"
            else:
                climate = "Temperate climate zone"
                
            summary += f"Climate Profile: {climate}\n"
            
            # Key concerns
            concerns = []
            if max_wind > 80:
                concerns.append("high wind events")
            if extreme_events > 3:
                concerns.append("frequent weather extremes")
            if avg_temp > 35:
                concerns.append("extreme heat")
            
            if concerns:
                summary += f"Primary Concerns: {', '.join(concerns)}"
            else:
                summary += "Generally stable weather patterns"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating intelligence summary: {e}")
            return "Weather intelligence summary unavailable"
    
    def _analyze_short_term_threats(self, forecast_data: Dict) -> List[Dict]:
        """Analyze immediate weather threats"""
        
        events = []
        try:
            if not forecast_data or 'hourly' not in forecast_data:
                return events
            
            hourly_df = pd.DataFrame(forecast_data['hourly'])
            if 'time' not in hourly_df.columns:
                return events
            
            hourly_df['time'] = pd.to_datetime(hourly_df['time'])
            
            # High wind events
            winds = pd.to_numeric(hourly_df.get('wind_speed_10m', []), errors='coerce').fillna(0)
            if winds.max() > 70:
                max_wind_idx = winds.idxmax()
                wind_time = hourly_df.iloc[max_wind_idx]['time']
                hours_ahead = (wind_time - datetime.now()).total_seconds() / 3600
                
                if hours_ahead > 0:
                    events.append({
                        'type': 'Severe Wind Event',
                        'timing': f"In {hours_ahead:.0f} hours",
                        'confidence': min(85, int(winds.max() * 0.9))
                    })
            
            # Heavy precipitation
            precip = pd.to_numeric(hourly_df.get('precipitation', []), errors='coerce').fillna(0)
            
            for i in range(len(precip) - 6):
                window_total = precip.iloc[i:i+6].sum()
                if window_total > 60:
                    event_start = hourly_df.iloc[i]['time']
                    hours_ahead = (event_start - datetime.now()).total_seconds() / 3600
                    
                    if hours_ahead > 0:
                        events.append({
                            'type': 'Flash Flood Risk',
                            'timing': f"Starting in {hours_ahead:.0f} hours",
                            'confidence': 75
                        })
                        break
            
            # Temperature extremes
            temps = pd.to_numeric(hourly_df.get('temperature_2m', []), errors='coerce').fillna(20)
            if temps.max() > 38:
                heat_idx = temps.idxmax()
                heat_time = hourly_df.iloc[heat_idx]['time']
                hours_ahead = (heat_time - datetime.now()).total_seconds() / 3600
                
                if hours_ahead > 0:
                    events.append({
                        'type': 'Extreme Heat Event',
                        'timing': f"Peak in {hours_ahead:.0f} hours",
                        'confidence': 90
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing short-term threats: {e}")
        
        return events
    
    def _analyze_seasonal_patterns(self, historical_data: Dict, location_name: str) -> List[Dict]:
        """Analyze seasonal weather patterns for long-term predictions"""
        
        events = []
        try:
            if not historical_data or 'extreme_events' not in historical_data:
                return events
            
            current_month = datetime.now().month
            location_lower = location_name.lower()
            
            # Hurricane season predictions
            if current_month in [5, 6, 7, 8, 9, 10] and any(term in location_lower for term in ['florida', 'gulf', 'atlantic', 'caribbean', 'miami']):
                events.append({
                    'type': 'Hurricane Season Activity',
                    'timeframe': 'June-November',
                    'probability': 70
                })
            
            # Tornado season predictions
            if current_month in [3, 4, 5] and any(term in location_lower for term in ['oklahoma', 'kansas', 'texas', 'plains', 'tornado']):
                events.append({
                    'type': 'Tornado Season Activity',
                    'timeframe': 'March-May',
                    'probability': 65
                })
            
            # Heat wave predictions
            if current_month in [6, 7, 8] and any(term in location_lower for term in ['phoenix', 'arizona', 'desert', 'southwest']):
                events.append({
                    'type': 'Extreme Heat Events',
                    'timeframe': 'Summer months',
                    'probability': 80
                })
            
            # Winter storm predictions
            if current_month in [11, 12, 1, 2] and any(term in location_lower for term in ['buffalo', 'great lakes', 'northern', 'snow']):
                events.append({
                    'type': 'Winter Storm Activity',
                    'timeframe': 'December-February',
                    'probability': 75
                })
        
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
        
        return events
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value is None:
                return default
            return float(value)
        except:
            return default

class EnhancedWatsonxIntegration:
    """Enhanced watsonx.ai integration with focused responses"""
    
    def __init__(self, original_integration):
        self.original = original_integration
        self.analyzer = EnhancedWeatherAnalyzer()
    
    def generate_focused_weather_analysis(self, weather_data: Dict, forecast_data: Dict, location_name: str) -> str:
        """Generate focused weather analysis with current conditions, forecast, predictions, and intelligence"""
        
        try:
            # Generate the four required sections
            current_conditions = self.analyzer.generate_current_conditions(forecast_data, location_name)
            
            forecast_summary = self.analyzer.generate_7day_forecast_summary(forecast_data, location_name)
            
            extreme_predictions = self.analyzer.predict_extreme_weather_events(forecast_data, weather_data, location_name)
            
            intelligence_summary = self.analyzer.generate_weather_intelligence_summary(forecast_data, weather_data, location_name)
            
            # Combine sections with clear separators
            response = f"{current_conditions}\n\n{forecast_summary}\n\n{extreme_predictions}\n\n{intelligence_summary}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating focused analysis: {e}")
            return f"Weather analysis unavailable for {location_name}: {str(e)}"

def enhance_rag_system_responses(rag_system):
    """Enhance existing RAG system with focused response generation"""
    
    if rag_system and rag_system.ai_integration:
        # Create enhanced integration
        enhanced_integration = EnhancedWatsonxIntegration(rag_system.ai_integration)
        
        # Store forecast data for analysis
        original_analyze = rag_system.analyze_location
        
        def enhanced_analyze_location(lat, lon, location_name, days_history=30):
            # Get both historical and forecast data
            historical_result = original_analyze(lat, lon, location_name, days_history)
            
            try:
                # Get fresh forecast data
                forecast_data = rag_system.weather_collector.get_current_forecast(lat, lon, days=7)
                
                # Generate focused analysis
                focused_analysis = enhanced_integration.generate_focused_weather_analysis(
                    historical_result['historical_analysis'], 
                    forecast_data, 
                    location_name
                )
                
                # Replace AI analysis with focused version
                historical_result['ai_analysis'] = focused_analysis
                
            except Exception as e:
                logger.error(f"Error in enhanced analysis: {e}")
            
            return historical_result
        
        # Replace the analyze method
        rag_system.analyze_location = enhanced_analyze_location
    
    return rag_system
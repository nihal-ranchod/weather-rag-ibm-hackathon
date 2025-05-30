#!/usr/bin/env python3
"""
Enhanced AI Response Generator for Weather RAG System
Clean, focused on predictions and long-term forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedWeatherAnalyzer:
    """Enhanced weather analysis with long-term prediction focus"""
    
    def __init__(self):
        self.severity_thresholds = {
            'hurricane': {'wind': 119, 'pressure_drop': 20},
            'severe_storm': {'wind': 93, 'precipitation': 25},
            'heat_wave': {'temperature': 35, 'duration': 3},
            'cold_snap': {'temperature': -18, 'duration': 2},
            'heavy_rain': {'precipitation': 50, 'rate': 25},
            'drought': {'precipitation': 5, 'duration': 14}
        }
        
        # Climate patterns for enhanced seasonal analysis
        self.climate_patterns = {
            'hurricane_season': {'months': [6, 7, 8, 9, 10, 11], 'regions': ['atlantic', 'gulf', 'caribbean', 'pacific']},
            'tornado_season': {'months': [3, 4, 5, 6], 'regions': ['plains', 'midwest', 'tornado alley']},
            'monsoon_season': {'months': [6, 7, 8, 9], 'regions': ['asia', 'india', 'southeast asia']},
            'typhoon_season': {'months': [5, 6, 7, 8, 9, 10, 11], 'regions': ['pacific', 'asia', 'philippines']},
            'wildfire_season': {'months': [6, 7, 8, 9, 10], 'regions': ['california', 'australia', 'mediterranean']},
            'blizzard_season': {'months': [12, 1, 2, 3], 'regions': ['northern', 'arctic', 'great lakes']}
        }
    
    def generate_7day_forecast_analysis(self, forecast_data: Dict, historical_data: Dict, location_name: str) -> str:
        """Generate detailed 7-day forecast with predictions"""
        
        if not forecast_data or 'daily' not in forecast_data:
            return "Forecast data unavailable for detailed analysis."
        
        try:
            daily_data = forecast_data['daily']
            df_daily = pd.DataFrame(daily_data)
            if 'time' in df_daily.columns:
                df_daily['time'] = pd.to_datetime(df_daily['time'])
            
            forecast_text = f"**7-DAY WEATHER INTELLIGENCE FOR {location_name.upper()}**\n\n"
            forecast_text += "**Daily Forecast & Risk Assessment:**\n\n"
            
            for i, row in df_daily.head(7).iterrows():
                date = row.get('time', datetime.now() + timedelta(days=i))
                if pd.isna(date):
                    date = datetime.now() + timedelta(days=i)
                
                day_name = date.strftime('%A')
                date_str = date.strftime('%b %d')
                
                temp_max = self._safe_float(row.get('temperature_2m_max'), 25)
                temp_min = self._safe_float(row.get('temperature_2m_min'), 15)
                precip = self._safe_float(row.get('precipitation_sum'), 0)
                wind_max = self._safe_float(row.get('wind_speed_10m_max'), 10)
                
                risk_level, risk_factors = self._assess_daily_risk(temp_max, temp_min, precip, wind_max)
                
                forecast_text += f"**{day_name}, {date_str}:**\n"
                forecast_text += f"• Temperature: {temp_min:.0f}°C to {temp_max:.0f}°C\n"
                forecast_text += f"• Precipitation: {precip:.1f}mm | Wind: {wind_max:.0f} km/h\n"
                forecast_text += f"• Risk Level: **{risk_level}**"
                
                if risk_factors:
                    forecast_text += f" - {', '.join(risk_factors)}"
                forecast_text += "\n\n"
            
            # Add weekly trends
            forecast_text += self._analyze_weekly_trends(df_daily, location_name)
            
            return forecast_text
            
        except Exception as e:
            logger.error(f"Error generating forecast analysis: {e}")
            return "Unable to generate detailed forecast analysis."
    
    def predict_long_term_weather_events(self, forecast_data: Dict, historical_data: Dict, location_name: str) -> str:
        """Enhanced long-term weather event predictions using historical patterns"""
        
        try:
            events_text = "**EXTENDED WEATHER EVENT PREDICTIONS**\n\n"
            
            # Immediate predictions (next 7 days)
            immediate_events = self._analyze_immediate_events(forecast_data)
            
            # Long-term predictions using historical patterns (30-90 days)
            long_term_events = self._analyze_long_term_patterns(historical_data, location_name)
            
            # Seasonal outlook (3-6 months)
            seasonal_outlook = self._analyze_seasonal_outlook(historical_data, location_name)
            
            if immediate_events:
                events_text += "**Next 7 Days:**\n"
                for event in immediate_events[:2]:
                    events_text += f"► **{event['event_type']}**\n"
                    events_text += f"  • Timing: {event['timing']}\n"
                    events_text += f"  • Probability: {event['probability']}%\n"
                    events_text += f"  • Peak Intensity: {event['peak_time']}\n"
                    events_text += f"  • Indicators: {event['indicators']}\n\n"
            
            if long_term_events:
                events_text += "**30-90 Day Outlook:**\n"
                for event in long_term_events:
                    events_text += f"► **{event['event_type']}**\n"
                    events_text += f"  • Expected Timeframe: {event['timeframe']}\n"
                    events_text += f"  • Historical Frequency: {event['frequency']}\n"
                    events_text += f"  • Confidence Level: {event['confidence']}\n\n"
            
            if seasonal_outlook:
                events_text += "**Seasonal Outlook (3-6 Months):**\n"
                events_text += seasonal_outlook + "\n\n"
            
            if not immediate_events and not long_term_events:
                events_text += "No major weather events predicted in available forecast period.\n"
                events_text += "Historical patterns suggest normal seasonal variations for this region.\n\n"
            
            return events_text
            
        except Exception as e:
            logger.error(f"Error predicting long-term events: {e}")
            return "Unable to generate extended weather event predictions."
    
    def generate_enhanced_seasonal_context(self, historical_data: Dict, location_name: str) -> str:
        """Generate enhanced seasonal context using historical patterns"""
        
        try:
            context_text = "**ENHANCED SEASONAL CONTEXT**\n\n"
            
            # Determine current season and location characteristics
            current_month = datetime.now().month
            location_lower = location_name.lower()
            
            # Analyze historical patterns by season
            seasonal_analysis = self._analyze_historical_by_season(historical_data)
            
            # Climate zone analysis
            climate_zone = self._determine_climate_zone(location_name, historical_data)
            context_text += f"**Climate Classification:** {climate_zone}\n\n"
            
            # Current seasonal risks
            current_risks = self._identify_current_seasonal_risks(current_month, location_lower)
            if current_risks:
                context_text += f"**Current Seasonal Risks:** {current_risks}\n\n"
            
            # Historical seasonal patterns
            if seasonal_analysis:
                context_text += "**Historical Seasonal Patterns:**\n"
                for season, data in seasonal_analysis.items():
                    if data['event_count'] > 0:
                        context_text += f"• {season}: {data['event_count']} extreme events, "
                        context_text += f"typical {data['dominant_type']} activity\n"
                context_text += "\n"
            
            # Long-term climate trends
            climate_trends = self._analyze_climate_trends(historical_data, location_name)
            if climate_trends:
                context_text += f"**Climate Trends:** {climate_trends}\n\n"
            
            return context_text
            
        except Exception as e:
            logger.error(f"Error generating seasonal context: {e}")
            return "Seasonal context analysis unavailable."
    
    def _analyze_immediate_events(self, forecast_data: Dict) -> List[Dict]:
        """Analyze immediate weather events in next 7 days"""
        
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
                        'event_type': 'Severe Wind Event',
                        'timing': f"In {hours_ahead:.0f} hours",
                        'probability': min(85, int(winds.max() * 0.9)),
                        'peak_time': wind_time.strftime('%A %I:%M %p'),
                        'indicators': f"Peak winds {winds.max():.0f} km/h"
                    })
            
            # Heavy precipitation events
            precip = pd.to_numeric(hourly_df.get('precipitation', []), errors='coerce').fillna(0)
            
            # Look for 6-hour precipitation totals
            for i in range(len(precip) - 6):
                window_total = precip.iloc[i:i+6].sum()
                if window_total > 60:  # Heavy rain threshold
                    event_start = hourly_df.iloc[i]['time']
                    hours_ahead = (event_start - datetime.now()).total_seconds() / 3600
                    
                    if hours_ahead > 0:
                        events.append({
                            'event_type': 'Flash Flood Risk',
                            'timing': f"Starting in {hours_ahead:.0f} hours",
                            'probability': 75,
                            'peak_time': event_start.strftime('%A %I:%M %p'),
                            'indicators': f"{window_total:.0f}mm in 6 hours"
                        })
                        break
            
            # Temperature extremes
            temps = pd.to_numeric(hourly_df.get('temperature_2m', []), errors='coerce').fillna(20)
            if temps.max() > 38:  # Extreme heat
                heat_idx = temps.idxmax()
                heat_time = hourly_df.iloc[heat_idx]['time']
                hours_ahead = (heat_time - datetime.now()).total_seconds() / 3600
                
                if hours_ahead > 0:
                    events.append({
                        'event_type': 'Extreme Heat Event',
                        'timing': f"Peak in {hours_ahead:.0f} hours",
                        'probability': 90,
                        'peak_time': heat_time.strftime('%A %I:%M %p'),
                        'indicators': f"Temperature reaching {temps.max():.0f}°C"
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing immediate events: {e}")
        
        return events
    
    def _analyze_long_term_patterns(self, historical_data: Dict, location_name: str) -> List[Dict]:
        """Analyze long-term weather patterns for 30-90 day predictions"""
        
        events = []
        try:
            if not historical_data or 'extreme_events' not in historical_data:
                return events
            
            extreme_events = historical_data['extreme_events']
            if not extreme_events:
                return events
            
            # Analyze event frequency by type and timing
            event_calendar = {}
            for event in extreme_events:
                try:
                    event_time = pd.to_datetime(event['timestamp'])
                    month = event_time.month
                    event_type = event['type']
                    
                    if month not in event_calendar:
                        event_calendar[month] = {}
                    if event_type not in event_calendar[month]:
                        event_calendar[month][event_type] = 0
                    event_calendar[month][event_type] += 1
                except:
                    continue
            
            # Predict based on historical patterns
            current_month = datetime.now().month
            
            # Look ahead 1-3 months
            for month_offset in range(1, 4):
                target_month = ((current_month + month_offset - 1) % 12) + 1
                
                if target_month in event_calendar:
                    for event_type, count in event_calendar[target_month].items():
                        if count >= 2:  # Significant historical occurrence
                            month_name = datetime(2024, target_month, 1).strftime('%B')
                            
                            # Calculate probability based on frequency
                            probability = min(80, count * 25)
                            
                            events.append({
                                'event_type': f'{event_type} Activity',
                                'timeframe': f'{month_name} (historically active period)',
                                'frequency': f'{count} events in historical data',
                                'confidence': f'{probability}%'
                            })
        
        except Exception as e:
            logger.error(f"Error analyzing long-term patterns: {e}")
        
        return events
    
    def _analyze_seasonal_outlook(self, historical_data: Dict, location_name: str) -> str:
        """Generate 3-6 month seasonal outlook"""
        
        try:
            current_month = datetime.now().month
            location_lower = location_name.lower()
            
            # Determine upcoming seasons
            if current_month in [12, 1, 2]:
                upcoming_season = "winter/spring transition"
                risk_factors = ["temperature extremes", "late winter storms"]
            elif current_month in [3, 4, 5]:
                upcoming_season = "spring/summer transition"
                risk_factors = ["severe thunderstorms", "tornado activity"]
            elif current_month in [6, 7, 8]:
                upcoming_season = "peak summer"
                risk_factors = ["heat waves", "tropical activity"]
            else:
                upcoming_season = "autumn/winter transition"
                risk_factors = ["tropical systems", "early winter weather"]
            
            outlook = f"Entering {upcoming_season} period. "
            
            # Location-specific seasonal risks
            if any(term in location_lower for term in ['florida', 'gulf', 'atlantic', 'caribbean']):
                if current_month in [5, 6, 7, 8, 9]:
                    outlook += "Hurricane season active through November. "
            elif any(term in location_lower for term in ['tornado', 'oklahoma', 'kansas', 'texas']):
                if current_month in [2, 3, 4, 5]:
                    outlook += "Peak tornado season approaching. "
            elif any(term in location_lower for term in ['california', 'west coast']):
                if current_month in [6, 7, 8, 9]:
                    outlook += "Wildfire season continues through fall. "
            
            outlook += f"Primary concerns: {', '.join(risk_factors)}."
            
            return outlook
            
        except Exception as e:
            logger.error(f"Error generating seasonal outlook: {e}")
            return "Seasonal outlook analysis unavailable."
    
    def _analyze_historical_by_season(self, historical_data: Dict) -> Dict:
        """Analyze historical data by season"""
        
        try:
            if not historical_data or 'extreme_events' not in historical_data:
                return {}
            
            seasons = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11]
            }
            
            seasonal_data = {}
            for season, months in seasons.items():
                seasonal_data[season] = {'event_count': 0, 'event_types': {}}
            
            for event in historical_data['extreme_events']:
                try:
                    event_time = pd.to_datetime(event['timestamp'])
                    month = event_time.month
                    event_type = event['type']
                    
                    for season, months in seasons.items():
                        if month in months:
                            seasonal_data[season]['event_count'] += 1
                            if event_type not in seasonal_data[season]['event_types']:
                                seasonal_data[season]['event_types'][event_type] = 0
                            seasonal_data[season]['event_types'][event_type] += 1
                            break
                except:
                    continue
            
            # Determine dominant event type for each season
            for season in seasonal_data:
                if seasonal_data[season]['event_types']:
                    dominant = max(seasonal_data[season]['event_types'], 
                                 key=seasonal_data[season]['event_types'].get)
                    seasonal_data[season]['dominant_type'] = dominant
                else:
                    seasonal_data[season]['dominant_type'] = 'none'
            
            return seasonal_data
            
        except Exception as e:
            logger.error(f"Error analyzing historical by season: {e}")
            return {}
    
    def _determine_climate_zone(self, location_name: str, historical_data: Dict) -> str:
        """Determine climate zone based on location and data"""
        
        location_lower = location_name.lower()
        stats = historical_data.get('statistics', {})
        avg_temp = stats.get('avg_temperature', 20)
        
        if any(term in location_lower for term in ['arctic', 'alaska', 'greenland']):
            return "Arctic Climate"
        elif any(term in location_lower for term in ['canada', 'russia', 'scandinavia']) and avg_temp < 10:
            return "Subarctic Climate"
        elif any(term in location_lower for term in ['tropical', 'equator', 'amazon', 'congo']):
            return "Tropical Climate"
        elif any(term in location_lower for term in ['desert', 'sahara', 'arizona', 'nevada']):
            return "Arid Climate"
        elif any(term in location_lower for term in ['mediterranean', 'california', 'chile']):
            return "Mediterranean Climate"
        elif avg_temp > 25:
            return "Subtropical Climate"
        elif avg_temp < 10:
            return "Continental Climate"
        else:
            return "Temperate Climate"
    
    def _identify_current_seasonal_risks(self, current_month: int, location_lower: str) -> str:
        """Identify current seasonal risks based on month and location"""
        
        risks = []
        
        # Hurricane/Typhoon season
        if current_month in [6, 7, 8, 9, 10, 11]:
            if any(term in location_lower for term in ['atlantic', 'gulf', 'caribbean', 'florida']):
                risks.append("Atlantic hurricane season")
            elif any(term in location_lower for term in ['pacific', 'asia', 'philippines', 'japan']):
                risks.append("Pacific typhoon season")
        
        # Tornado season
        if current_month in [3, 4, 5, 6]:
            if any(term in location_lower for term in ['tornado', 'oklahoma', 'kansas', 'plains']):
                risks.append("peak tornado season")
        
        # Wildfire season
        if current_month in [6, 7, 8, 9, 10]:
            if any(term in location_lower for term in ['california', 'australia', 'mediterranean']):
                risks.append("wildfire season")
        
        # Winter storms
        if current_month in [12, 1, 2, 3]:
            if any(term in location_lower for term in ['northern', 'canada', 'great lakes']):
                risks.append("winter storm season")
        
        # Monsoon season
        if current_month in [6, 7, 8, 9]:
            if any(term in location_lower for term in ['india', 'bangladesh', 'myanmar']):
                risks.append("monsoon season")
        
        return ", ".join(risks) if risks else "normal seasonal patterns"
    
    def _analyze_climate_trends(self, historical_data: Dict, location_name: str) -> str:
        """Analyze long-term climate trends"""
        
        try:
            stats = historical_data.get('statistics', {})
            extreme_events = stats.get('total_extreme_events', 0)
            max_temp = stats.get('max_temperature', 20)
            total_precip = stats.get('total_precipitation', 0)
            
            trends = []
            
            if extreme_events > 5:
                trends.append("increased extreme weather frequency")
            if max_temp > 35:
                trends.append("elevated temperature extremes")
            if total_precip > 200:
                trends.append("above-average precipitation patterns")
            elif total_precip < 50:
                trends.append("below-average precipitation patterns")
            
            if trends:
                return f"Recent patterns show {', '.join(trends)}"
            else:
                return "stable climate patterns within normal ranges"
                
        except Exception as e:
            logger.error(f"Error analyzing climate trends: {e}")
            return "climate trend analysis unavailable"
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float"""
        try:
            if pd.isna(value) or value is None:
                return default
            return float(value)
        except:
            return default
    
    def _assess_daily_risk(self, temp_max, temp_min, precip, wind_max):
        """Assess risk level for a single day"""
        risk_score = 0
        risk_factors = []
        
        if temp_max > 35:
            risk_score += 3
            risk_factors.append("extreme heat")
        elif temp_min < -10:
            risk_score += 3
            risk_factors.append("extreme cold")
        
        if wind_max > 60:
            risk_score += 3
            risk_factors.append("strong winds")
        elif wind_max > 40:
            risk_score += 1
            risk_factors.append("moderate winds")
        
        if precip > 30:
            risk_score += 2
            risk_factors.append("heavy rain")
        elif precip > 10:
            risk_score += 1
            risk_factors.append("rain")
        
        if risk_score >= 5:
            return "HIGH", risk_factors
        elif risk_score >= 2:
            return "MODERATE", risk_factors
        else:
            return "LOW", risk_factors
    
    def _analyze_weekly_trends(self, df_daily: pd.DataFrame, location_name: str) -> str:
        """Analyze trends across the 7-day period"""
        
        try:
            trends = "**Weekly Weather Trends:**\n\n"
            
            if 'temperature_2m_max' in df_daily.columns:
                temps = pd.to_numeric(df_daily['temperature_2m_max'], errors='coerce').dropna()
                if len(temps) >= 3:
                    temp_trend = "warming" if temps.iloc[-1] > temps.iloc[0] else "cooling"
                    temp_change = abs(temps.iloc[-1] - temps.iloc[0])
                    trends += f"• Temperature Pattern: {temp_trend.title()} trend with {temp_change:.1f}°C change\n"
            
            if 'precipitation_sum' in df_daily.columns:
                precip = pd.to_numeric(df_daily['precipitation_sum'], errors='coerce').dropna()
                if len(precip) >= 3:
                    wet_days = (precip > 1).sum()
                    total_precip = precip.sum()
                    trends += f"• Precipitation Outlook: {wet_days} wet days expected, {total_precip:.1f}mm total\n"
            
            if 'wind_speed_10m_max' in df_daily.columns:
                winds = pd.to_numeric(df_daily['wind_speed_10m_max'], errors='coerce').dropna()
                if len(winds) >= 3:
                    max_wind_day = winds.idxmax()
                    date_of_max = df_daily.iloc[max_wind_day]['time']
                    if pd.notna(date_of_max):
                        day_name = date_of_max.strftime('%A')
                        trends += f"• Wind Forecast: Peak winds expected {day_name} ({winds.max():.0f} km/h)\n"
            
            return trends + "\n"
            
        except Exception as e:
            logger.error(f"Error analyzing weekly trends: {e}")
            return ""

class EnhancedWatsonxIntegration:
    """Enhanced watsonx.ai integration with cleaner responses"""
    
    def __init__(self, original_integration):
        self.original = original_integration
        self.analyzer = EnhancedWeatherAnalyzer()
    
    def generate_enhanced_weather_analysis(self, weather_data: Dict, predictions: List[Dict], location_name: str = None) -> str:
        """Generate enhanced AI analysis focused on predictions"""
        
        location = location_name if location_name else "the analyzed location"
        
        # Get enhanced forecast data if available
        forecast_data = getattr(self.original, '_last_forecast_data', {})
        historical_data = weather_data
        
        # Generate the main sections
        forecast_analysis = self.analyzer.generate_7day_forecast_analysis(forecast_data, historical_data, location)
        
        event_predictions = self.analyzer.predict_long_term_weather_events(forecast_data, historical_data, location)
        
        seasonal_context = self.analyzer.generate_enhanced_seasonal_context(historical_data, location)
        
        # Combine sections
        enhanced_response = f"{forecast_analysis}\n{event_predictions}\n{seasonal_context}"
        
        # Add AI enhancement if available
        if self.original and self.original.is_initialized:
            try:
                ai_prompt = f"""Provide meteorological analysis for {location}:

Current Analysis Summary: {enhanced_response[:400]}...

Provide expert insights on:
1. Atmospheric dynamics causing these patterns
2. Unusual weather signatures or anomalies
3. Regional climate considerations
4. Confidence levels for predictions

Keep response under 200 words, focus on technical meteorological insights."""

                ai_enhancement = self.original.model.generate_text(ai_prompt)
                if ai_enhancement and len(ai_enhancement.strip()) > 30:
                    enhanced_response += f"\n**WEATHER INTELLIGENCE ANALYSIS:**\n{ai_enhancement.strip()}"
            
            except Exception as e:
                logger.error(f"AI enhancement failed: {e}")
        
        return enhanced_response

def enhance_rag_system_responses(rag_system):
    """Enhance existing RAG system with better response generation"""
    
    if rag_system and rag_system.ai_integration:
        # Create enhanced integration
        enhanced_integration = EnhancedWatsonxIntegration(rag_system.ai_integration)
        
        # Store original method and replace
        rag_system.ai_integration.generate_weather_analysis = enhanced_integration.generate_enhanced_weather_analysis
        
        # Enhanced analyze method to store forecast data
        original_analyze = rag_system.analyze_location
        
        def enhanced_analyze_location(lat, lon, location_name, days_history=30):
            result = original_analyze(lat, lon, location_name, days_history)
            
            try:
                forecast_data = rag_system.weather_collector.get_current_forecast(lat, lon, days=7)
                rag_system.ai_integration._last_forecast_data = forecast_data
            except:
                pass
            
            return result
        
        rag_system.analyze_location = enhanced_analyze_location
    
    return rag_system
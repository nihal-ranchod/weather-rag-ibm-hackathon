#!/usr/bin/env python3
"""
Test script to verify the Weather RAG System is working properly
Run this to diagnose issues with your system
"""

import sys
import os
import requests
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all imports work"""
    print("🔍 Testing imports...")
    
    try:
        import pandas
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        from weather_rag_system import WeatherDataCollector, ExtremeWeatherDetector, ExtremeWeatherRAGSystem
        print("✅ Weather RAG system imports successful")
    except ImportError as e:
        print(f"❌ Weather RAG system import failed: {e}")
        print("   Make sure weather_rag_system.py is in the src/ directory")
        return False
    
    try:
        from config import WATSONX_API_KEY, WATSONX_PROJECT_ID
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        print("   Make sure config.py is in the src/ directory")
        return False
    
    return True

def test_weather_api():
    """Test weather API connectivity"""
    print("\n🌐 Testing Weather API connectivity...")
    
    try:
        from weather_rag_system import WeatherDataCollector
        collector = WeatherDataCollector()
        
        # Test forecast API
        print("   Testing forecast API...")
        forecast = collector.get_current_forecast(40.7128, -74.0060, days=1)
        
        if forecast and forecast.get('hourly'):
            print("✅ Weather forecast API working")
            print(f"   Retrieved {len(forecast['hourly']['time'])} hourly forecasts")
            return True
        else:
            print("⚠️ Weather API returned empty/invalid data")
            print("   System will use mock data")
            return False
            
    except Exception as e:
        print(f"❌ Weather API test failed: {e}")
        return False

def test_location_service():
    """Test location geocoding"""
    print("\n📍 Testing location service...")
    
    try:
        # Test Nominatim directly
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": "Miami, FL",
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "WeatherRAGSystemTest/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print("✅ Location geocoding service working")
                print(f"   Miami coordinates: {data[0]['lat']}, {data[0]['lon']}")
                return True
        
        print("⚠️ Geocoding service issues detected")
        return False
        
    except Exception as e:
        print(f"❌ Location service test failed: {e}")
        return False

def test_watsonx_credentials():
    """Test watsonx.ai credentials"""
    print("\n🤖 Testing watsonx.ai credentials...")
    
    try:
        from config import WATSONX_API_KEY, WATSONX_PROJECT_ID
        
        if WATSONX_API_KEY == 'your_api_key_here':
            print("⚠️ watsonx.ai API key not configured")
            print("   Update WATSONX_API_KEY in src/config.py or environment variable")
            return False
        
        if WATSONX_PROJECT_ID == 'your_project_id_here':
            print("⚠️ watsonx.ai Project ID not configured")
            print("   Update WATSONX_PROJECT_ID in src/config.py or environment variable")
            return False
        
        print("✅ watsonx.ai credentials are configured")
        print(f"   API Key: {WATSONX_API_KEY[:20]}...")
        print(f"   Project ID: {WATSONX_PROJECT_ID}")
        
        # Try to initialize the system
        try:
            from weather_rag_system import ExtremeWeatherRAGSystem
            print("   Testing AI system initialization...")
            
            system = ExtremeWeatherRAGSystem(
                watsonx_api_key=WATSONX_API_KEY,
                watsonx_project_id=WATSONX_PROJECT_ID
            )
            
            if system.ai_integration.is_initialized:
                print("✅ watsonx.ai system initialized successfully")
                return True
            else:
                print("❌ watsonx.ai initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ watsonx.ai initialization error: {e}")
            return False
        
    except Exception as e:
        print(f"❌ watsonx.ai credential test failed: {e}")
        return False

def test_full_system():
    """Test the complete system end-to-end"""
    print("\n🔧 Testing complete system...")
    
    try:
        from weather_rag_system import ExtremeWeatherRAGSystem
        from config import WATSONX_API_KEY, WATSONX_PROJECT_ID
        
        # Initialize system
        system = ExtremeWeatherRAGSystem(
            watsonx_api_key=WATSONX_API_KEY,
            watsonx_project_id=WATSONX_PROJECT_ID
        )
        
        # Test quick analysis
        print("   Testing quick analysis...")
        result = system.quick_analysis(25.7617, -80.1918, "Miami, FL")
        
        if result and not result.startswith("❌"):
            print("✅ Quick analysis working")
            print(f"   Result preview: {result[:100]}...")
        else:
            print("⚠️ Quick analysis issues detected")
            print(f"   Result: {result}")
        
        # Test full analysis (short version)
        print("   Testing basic analysis components...")
        forecast = system.weather_collector.get_current_forecast(25.7617, -80.1918, days=1)
        predictions = system.detector.predict_extreme_events(forecast, {})
        
        print(f"✅ System test completed")
        print(f"   Forecast data: {'Available' if forecast else 'Mock data'}")
        print(f"   Predictions: {len(predictions)} events detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Weather RAG System Diagnostic Test")
    print("=" * 50)
    
    results = {
        "imports": test_imports(),
        "weather_api": test_weather_api(),
        "location_service": test_location_service(),
        "watsonx_credentials": test_watsonx_credentials(),
        "full_system": test_full_system()
    }
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<30} {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system should work perfectly.")
        print("\n🚀 You can now run: python app.py")
    elif passed >= 3:
        print("⚠️ Most tests passed. System will work with some limitations.")
        print("\n🔧 Recommendations:")
        if not results["watsonx_credentials"]:
            print("   • Configure watsonx.ai credentials for full AI features")
        if not results["weather_api"]:
            print("   • Check internet connection for live weather data")
        print("\n🚀 You can still run: python app.py")
    else:
        print("❌ Multiple test failures detected.")
        print("\n🔧 Required fixes:")
        if not results["imports"]:
            print("   • Install missing dependencies: pip install -r requirements.txt")
        if not results["weather_api"]:
            print("   • Check internet connection")
        if not results["location_service"]:
            print("   • Verify network access to geocoding services")
        print("\n   Fix these issues before running the application.")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    
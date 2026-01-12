"""
Simple local test for EcoHome - WITHOUT ChromaDB/RAG
Tests database and SQL query tools only
"""
import os
import sys
from datetime import datetime, timedelta

# Set API key
os.environ['OPENAI_API_KEY'] = 'sk-proj-YOUR-KEY-HERE'  # Replace with your key

print("="*60)
print("EcoHome Local Test (Simple - No RAG)")
print("="*60)

# Step 1: Test Database Setup
print("\n1. Testing Database Setup...")
try:
    from models.energy import init_db, EnergyUsage, SolarGeneration
    from models.energy import get_session
    from sqlalchemy import inspect
    
    db_path = "data/energy_data.db"  # Use same path as tools
    engine = init_db(db_path)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("✓ Database initialized")
    print(f"✓ Tables created: {tables}")
    
    # Add test data
    with get_session(db_path) as session:
        # Add some energy usage records
        base_time = datetime.now() - timedelta(days=7)
        for i in range(10):
            usage = EnergyUsage(
                timestamp=base_time + timedelta(hours=i),
                device_type='hvac',
                device_name='Central AC',
                energy_kwh=2.5,
                cost_usd=0.30,
                price_per_kwh=0.12
            )
            session.add(usage)
        
        # Add solar generation
        for i in range(10):
            solar = SolarGeneration(
                timestamp=base_time + timedelta(hours=i),
                generation_kwh=3.5,
                weather_condition='sunny',
                temperature_f=75.0,
                cloud_cover_percent=10.0
            )
            session.add(solar)
        
        session.commit()
        print(f"✓ Added 10 energy usage records")
        print(f"✓ Added 10 solar generation records")
        
except Exception as e:
    print(f"✗ Database setup failed: {e}")
    sys.exit(1)

# Step 2: Test SQL Query Tools
print("\n2. Testing SQL Query Tools...")
try:
    from tools import query_energy_usage, query_solar_generation
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    energy_result = query_energy_usage.invoke({"start_date": start_date, "end_date": end_date})
    print(f"✓ Energy query returned: {len(energy_result)} characters")
    print(f"  Preview: {energy_result[:200]}...")
    
    solar_result = query_solar_generation.invoke({"start_date": start_date, "end_date": end_date})
    print(f"✓ Solar query returned: {len(solar_result)} characters")
    print(f"  Preview: {solar_result[:200]}...")
    
except Exception as e:
    print(f"✗ Tool testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test Mock API Tools
print("\n3. Testing Mock API Tools...")
try:
    from tools import get_weather_forecast, get_electricity_prices
    
    weather = get_weather_forecast.invoke({"location": "San Francisco", "days": 5})
    print(f"✓ Weather forecast: {weather[:150]}...")
    
    prices = get_electricity_prices.invoke({"start_date": start_date, "end_date": end_date})
    print(f"✓ Electricity prices: {prices[:150]}...")
    
except Exception as e:
    print(f"✗ Mock API tools failed: {e}")
    sys.exit(1)

# Step 4: Test Agent (without RAG)
print("\n4. Testing Agent (SQL tools only)...")
try:
    from agent import create_agent
    
    # Create agent
    agent = create_agent()
    print("✓ Agent created successfully")
    
    # Test query
    config = {"configurable": {"thread_id": "test-1"}}
    test_query = "What was my total energy usage last week?"
    
    print(f"\nQuery: {test_query}")
    print("Response:")
    
    response = agent.app.invoke(
        {"messages": [("user", test_query)]},
        config=config
    )
    
    final_message = response["messages"][-1].content
    print(final_message)
    print("\n✓ Agent responded successfully!")
    
except Exception as e:
    print(f"✗ Agent test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nThe SQL database and agent are working!")
print("Note: RAG/ChromaDB not tested (Windows compatibility issues)")
print("For full RAG testing, use Google Colab instead.")

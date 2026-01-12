"""
EcoHome Energy Advisor Tools
Tools for weather forecasts, electricity pricing, database queries, and RAG search
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from models.energy import EnergyUsage, SolarGeneration, get_session


# Global variable for vector store (initialized once)
_vector_store = None


def initialize_vector_store(persist_directory: str = "./chroma_db") -> Chroma:
    """Initialize the RAG vector store from knowledge base documents."""
    global _vector_store
    
    if _vector_store is not None:
        return _vector_store
    
    # Load documents from knowledge base
    loader = DirectoryLoader(
        "./data/documents/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    _vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return _vector_store


@tool
def get_weather_forecast(location: str, days: int = 3) -> str:
    """
    Get weather forecast for a location.
    
    Args:
        location: City or location name (e.g., "San Francisco")
        days: Number of days to forecast (1-7, default 3)
    
    Returns:
        Weather forecast as formatted string with temperature, conditions, and solar potential
    """
    # Simulate weather forecast (in production, would call real weather API)
    days = min(max(days, 1), 7)  # Clamp between 1-7 days
    
    forecast_data = []
    current_date = datetime.now()
    
    # Simulated weather conditions
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Overcast", "Rainy"]
    
    for i in range(days):
        date = current_date + timedelta(days=i)
        condition = random.choice(conditions)
        high_temp = random.randint(65, 95)
        low_temp = high_temp - random.randint(15, 25)
        
        # Solar potential based on conditions
        solar_potential_map = {
            "Sunny": "Excellent (90-100%)",
            "Partly Cloudy": "Good (70-90%)",
            "Cloudy": "Moderate (40-70%)",
            "Overcast": "Low (20-40%)",
            "Rainy": "Poor (10-20%)"
        }
        solar_potential = solar_potential_map.get(condition, "Unknown")
        
        forecast_data.append({
            "date": date.strftime("%A, %B %d"),
            "condition": condition,
            "high": high_temp,
            "low": low_temp,
            "solar_potential": solar_potential
        })
    
    # Format output
    result = f"Weather Forecast for {location} ({days} days):\n\n"
    for day in forecast_data:
        result += f"{day['date']}:\n"
        result += f"  Conditions: {day['condition']}\n"
        result += f"  Temperature: {day['low']}°F - {day['high']}°F\n"
        result += f"  Solar Potential: {day['solar_potential']}\n\n"
    
    # Add energy recommendations based on forecast
    if any(d["condition"] == "Sunny" for d in forecast_data):
        result += "Recommendations:\n"
        result += "- Excellent solar generation expected - schedule high-energy tasks during midday\n"
        result += "- Pre-cool home during morning if temperatures will be high\n"
        result += "- Charge EV and battery storage during peak solar hours\n"
    elif all(d["condition"] in ["Cloudy", "Overcast", "Rainy"] for d in forecast_data):
        result += "Recommendations:\n"
        result += "- Limited solar generation expected - rely on battery storage and grid\n"
        result += "- Consider charging battery from grid during off-peak hours\n"
        result += "- Postpone non-essential high-energy tasks if possible\n"
    
    return result


@tool
def get_electricity_prices(date: Optional[str] = None) -> str:
    """
    Get electricity pricing information for time-of-use rates.
    
    Args:
        date: Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Electricity pricing schedule with rates and time periods
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Simulated TOU rates (would come from utility API in production)
    weekday_rates = {
        "Off-Peak (12 AM - 6 AM)": "$0.12/kWh",
        "Partial-Peak (6 AM - 4 PM)": "$0.22/kWh",
        "Peak (4 PM - 9 PM)": "$0.42/kWh",
        "Partial-Peak (9 PM - 12 AM)": "$0.22/kWh"
    }
    
    weekend_rates = {
        "Off-Peak (All Day)": "$0.12/kWh"
    }
    
    # Determine if date is weekend
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    is_weekend = date_obj.weekday() >= 5
    
    rates = weekend_rates if is_weekend else weekday_rates
    day_type = "Weekend" if is_weekend else "Weekday"
    
    result = f"Electricity Rates for {date} ({day_type}):\n\n"
    
    for period, rate in rates.items():
        result += f"{period}: {rate}\n"
    
    result += "\nEnergy Saving Tips:\n"
    if not is_weekend:
        result += "- Avoid high-energy activities during peak hours (4 PM - 9 PM)\n"
        result += "- Run dishwasher, laundry, and EV charging during off-peak hours (12 AM - 6 AM)\n"
        result += "- Use battery storage to avoid peak rate charges\n"
        result += "- Pre-cool home before peak period using solar or partial-peak power\n"
    else:
        result += "- Great day to run high-energy appliances with off-peak rates all day\n"
        result += "- Perfect time for laundry, dishwasher, and other heavy usage\n"
        result += "- Still prioritize solar generation during daylight hours\n"
    
    # Add current time recommendation
    current_hour = datetime.now().hour
    if not is_weekend:
        if 0 <= current_hour < 6:
            result += f"\nCurrent Rate (Off-Peak): $0.12/kWh - Excellent time for high-energy tasks!\n"
        elif 6 <= current_hour < 16:
            result += f"\nCurrent Rate (Partial-Peak): $0.22/kWh - Moderate cost, prefer solar power if available.\n"
        elif 16 <= current_hour < 21:
            result += f"\nCurrent Rate (Peak): $0.42/kWh - AVOID high-energy tasks if possible!\n"
        else:
            result += f"\nCurrent Rate (Partial-Peak): $0.22/kWh - Moderate cost, consider waiting for off-peak.\n"
    else:
        result += f"\nCurrent Rate (Off-Peak): $0.12/kWh - Great rates all day!\n"
    
    return result


@tool
def query_energy_usage(
    start_date: str,
    end_date: str,
    aggregate_by: str = "day"
) -> str:
    """
    Query energy usage data from database for a date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        aggregate_by: Aggregation level - "hour", "day", or "month"
    
    Returns:
        Energy usage statistics and insights
    """
    session = get_session()
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Query energy usage
        query = session.query(
            func.sum(EnergyUsage.total_kwh).label("total_usage"),
            func.avg(EnergyUsage.total_kwh).label("avg_usage"),
            func.max(EnergyUsage.total_kwh).label("max_usage"),
            func.min(EnergyUsage.total_kwh).label("min_usage"),
            func.avg(EnergyUsage.hvac_kwh).label("avg_hvac"),
            func.avg(EnergyUsage.appliances_kwh).label("avg_appliances"),
            func.avg(EnergyUsage.ev_charging_kwh).label("avg_ev"),
            func.avg(EnergyUsage.other_kwh).label("avg_other")
        ).filter(
            and_(
                EnergyUsage.timestamp >= start,
                EnergyUsage.timestamp <= end
            )
        )
        
        result = query.first()
        
        if result.total_usage is None:
            return f"No energy usage data found for {start_date} to {end_date}"
        
        # Calculate days in range
        days = (end - start).days + 1
        
        # Format output
        output = f"Energy Usage Summary ({start_date} to {end_date}):\n\n"
        output += f"Total Usage: {result.total_usage:.2f} kWh over {days} days\n"
        output += f"Average Daily Usage: {result.avg_usage:.2f} kWh\n"
        output += f"Peak Daily Usage: {result.max_usage:.2f} kWh\n"
        output += f"Minimum Daily Usage: {result.min_usage:.2f} kWh\n\n"
        
        output += "Usage Breakdown by Category:\n"
        output += f"  HVAC: {result.avg_hvac:.2f} kWh/day ({result.avg_hvac/result.avg_usage*100:.1f}%)\n"
        output += f"  Appliances: {result.avg_appliances:.2f} kWh/day ({result.avg_appliances/result.avg_usage*100:.1f}%)\n"
        output += f"  EV Charging: {result.avg_ev:.2f} kWh/day ({result.avg_ev/result.avg_usage*100:.1f}%)\n"
        output += f"  Other: {result.avg_other:.2f} kWh/day ({result.avg_other/result.avg_usage*100:.1f}%)\n\n"
        
        # Add insights
        output += "Insights:\n"
        
        # HVAC insights
        hvac_pct = result.avg_hvac / result.avg_usage * 100
        if hvac_pct > 50:
            output += f"- HVAC is {hvac_pct:.0f}% of usage - consider optimizing thermostat settings\n"
        elif hvac_pct > 40:
            output += f"- HVAC usage at {hvac_pct:.0f}% is typical for many homes\n"
        else:
            output += f"- HVAC usage at {hvac_pct:.0f}% is relatively low - good job!\n"
        
        # EV insights
        ev_pct = result.avg_ev / result.avg_usage * 100
        if ev_pct > 30:
            output += f"- High EV charging at {ev_pct:.0f}% - ensure charging during off-peak or solar hours\n"
        elif ev_pct > 0:
            output += f"- EV charging at {ev_pct:.0f}% of usage - monitor charging schedules\n"
        
        # Daily usage insights
        if result.avg_usage > 40:
            output += f"- Average daily usage of {result.avg_usage:.1f} kWh is above typical (30-35 kWh)\n"
            output += "  Consider energy audit to identify savings opportunities\n"
        elif result.avg_usage < 25:
            output += f"- Excellent! Average usage of {result.avg_usage:.1f} kWh is below typical\n"
        
        return output
        
    except Exception as e:
        return f"Error querying energy usage: {str(e)}"
    finally:
        session.close()


@tool
def query_solar_generation(
    start_date: str,
    end_date: str
) -> str:
    """
    Query solar generation data from database for a date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        Solar generation statistics and insights
    """
    session = get_session()
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Query solar generation
        query = session.query(
            func.sum(SolarGeneration.generated_kwh).label("total_generation"),
            func.avg(SolarGeneration.generated_kwh).label("avg_generation"),
            func.max(SolarGeneration.generated_kwh).label("max_generation"),
            func.min(SolarGeneration.generated_kwh).label("min_generation"),
            func.avg(SolarGeneration.self_consumed_kwh).label("avg_self_consumed"),
            func.avg(SolarGeneration.exported_kwh).label("avg_exported"),
            func.avg(SolarGeneration.battery_stored_kwh).label("avg_stored")
        ).filter(
            and_(
                SolarGeneration.timestamp >= start,
                SolarGeneration.timestamp <= end
            )
        )
        
        result = query.first()
        
        if result.total_generation is None:
            return f"No solar generation data found for {start_date} to {end_date}"
        
        # Also get energy usage for comparison
        usage_query = session.query(
            func.avg(EnergyUsage.total_kwh).label("avg_usage")
        ).filter(
            and_(
                EnergyUsage.timestamp >= start,
                EnergyUsage.timestamp <= end
            )
        )
        usage_result = usage_query.first()
        
        # Calculate days in range
        days = (end - start).days + 1
        
        # Format output
        output = f"Solar Generation Summary ({start_date} to {end_date}):\n\n"
        output += f"Total Generation: {result.total_generation:.2f} kWh over {days} days\n"
        output += f"Average Daily Generation: {result.avg_generation:.2f} kWh\n"
        output += f"Peak Daily Generation: {result.max_generation:.2f} kWh\n"
        output += f"Minimum Daily Generation: {result.min_generation:.2f} kWh\n\n"
        
        output += "Solar Energy Distribution:\n"
        output += f"  Self-Consumed: {result.avg_self_consumed:.2f} kWh/day ({result.avg_self_consumed/result.avg_generation*100:.1f}%)\n"
        output += f"  Exported to Grid: {result.avg_exported:.2f} kWh/day ({result.avg_exported/result.avg_generation*100:.1f}%)\n"
        output += f"  Stored in Battery: {result.avg_stored:.2f} kWh/day ({result.avg_stored/result.avg_generation*100:.1f}%)\n\n"
        
        # Add insights
        output += "Insights:\n"
        
        # Self-consumption rate
        self_consumption_rate = result.avg_self_consumed / result.avg_generation * 100
        if self_consumption_rate < 50:
            output += f"- Self-consumption rate of {self_consumption_rate:.0f}% is low\n"
            output += "  Tip: Increase battery storage or shift usage to solar hours\n"
        elif self_consumption_rate < 70:
            output += f"- Self-consumption rate of {self_consumption_rate:.0f}% is moderate\n"
            output += "  Tip: Consider adding battery storage to capture more solar\n"
        else:
            output += f"- Excellent self-consumption rate of {self_consumption_rate:.0f}%!\n"
        
        # Solar offset
        if usage_result.avg_usage:
            solar_offset = (result.avg_generation / usage_result.avg_usage) * 100
            output += f"- Solar offsets {solar_offset:.0f}% of your daily energy usage\n"
            if solar_offset >= 100:
                output += "  Excellent! You're generating more than you use\n"
            elif solar_offset >= 70:
                output += "  Great solar coverage - consider battery to increase self-consumption\n"
            else:
                output += "  Consider expanding solar array or reducing usage\n"
        
        # Export insights
        export_rate = result.avg_exported / result.avg_generation * 100
        if export_rate > 40:
            output += f"- High export rate of {export_rate:.0f}% - consider adding battery storage\n"
            output += "  Storing excess solar can provide greater value than export rates\n"
        
        # Storage insights  
        storage_rate = result.avg_stored / result.avg_generation * 100
        if storage_rate > 30:
            output += f"- Good battery utilization at {storage_rate:.0f}% of generation\n"
        elif storage_rate > 0:
            output += f"- Battery storage at {storage_rate:.0f}% - room to optimize storage strategy\n"
        
        return output
        
    except Exception as e:
        return f"Error querying solar generation: {str(e)}"
    finally:
        session.close()


@tool
def search_energy_tips(query: str, num_results: int = 3) -> str:
    """
    Search the knowledge base for energy-saving tips and recommendations.
    Uses RAG (Retrieval Augmented Generation) to find relevant information.
    
    Args:
        query: Search query or question about energy saving
        num_results: Number of relevant tips to return (default 3)
    
    Returns:
        Relevant energy-saving tips from the knowledge base
    """
    try:
        # Initialize vector store if needed
        vector_store = initialize_vector_store()
        
        # Search for relevant documents
        results = vector_store.similarity_search(query, k=num_results)
        
        if not results:
            return "No relevant tips found. Please try a different query."
        
        # Format output
        output = f"Energy-Saving Tips (found {len(results)} relevant results):\n\n"
        
        for i, doc in enumerate(results, 1):
            # Extract source document name
            source = doc.metadata.get("source", "Unknown")
            source_name = source.split("/")[-1] if "/" in source else source.split("\\")[-1]
            
            output += f"{i}. From {source_name}:\n"
            output += f"{doc.page_content}\n\n"
        
        return output
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


# Tool list for agent
ECOHOME_TOOLS = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    search_energy_tips
]


if __name__ == "__main__":
    # Test tools
    print("Testing EcoHome Tools\n" + "="*50 + "\n")
    
    # Test weather forecast
    print("1. Weather Forecast:")
    print(get_weather_forecast.invoke({"location": "San Francisco", "days": 3}))
    print("\n" + "="*50 + "\n")
    
    # Test electricity prices
    print("2. Electricity Prices:")
    print(get_electricity_prices.invoke({}))
    print("\n" + "="*50 + "\n")
    
    # Test search (requires initialization)
    print("3. Energy Tips Search:")
    try:
        print(search_energy_tips.invoke({"query": "how to reduce HVAC costs", "num_results": 2}))
    except Exception as e:
        print(f"Error: {e}")
    print("\n" + "="*50 + "\n")

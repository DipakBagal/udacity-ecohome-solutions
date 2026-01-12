"""
EcoHome Energy Advisor Tools
Tools for weather forecasts, electricity pricing, database queries, and RAG search
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import pickle
import os
import re
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from models.energy import EnergyUsage, SolarGeneration, get_session


# Global variable for vector store (initialized once)
_vector_store = None


class SimpleVectorStore:
    """Simulated vector store using keyword-based search (no API calls needed)"""
    
    def __init__(self, persist_directory=None):
        self.chunks = []
        self.keywords_list = []
        self.persist_directory = persist_directory
        
    def from_documents(self, documents, persist_directory=None):
        """Create vector store from documents"""
        store = SimpleVectorStore(persist_directory)
        store.chunks = documents
        
        # Extract keywords from all chunks
        for doc in documents:
            keywords = self._extract_keywords(doc.page_content)
            store.keywords_list.append(keywords)
        
        # Persist if directory specified
        if persist_directory:
            store.persist()
        
        return store
    
    def persist(self):
        """Save to disk"""
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            data = {
                'chunks': self.chunks,
                'keywords': self.keywords_list
            }
            with open(os.path.join(self.persist_directory, 'vectorstore.pkl'), 'wb') as f:
                pickle.dump(data, f)
    
    def load(self, persist_directory):
        """Load from disk"""
        self.persist_directory = persist_directory
        
        with open(os.path.join(persist_directory, 'vectorstore.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.keywords_list = data['keywords']
        
        return self
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        """Search for similar documents using keyword matching"""
        query_keywords = self._extract_keywords(query)
        
        # Calculate similarity scores
        similarities = []
        for i, doc_keywords in enumerate(self.keywords_list):
            # Calculate overlap between query and document keywords
            overlap = len(query_keywords & doc_keywords)
            # Normalize by query keywords
            score = overlap / len(query_keywords) if query_keywords else 0
            similarities.append((score, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        top_k = similarities[:k]
        
        return [self.chunks[idx] for _, idx in top_k]
    
    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extract important keywords from text"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'if', 'than', 'so',
            'up', 'out', 'about', 'into', 'through', 'during', 'before', 'after'
        }
        
        # Filter out stop words and short words
        keywords = {w for w in words if w not in stop_words and len(w) > 3}
        
        return keywords


def initialize_vector_store(persist_directory: str = "./chroma_db"):
    """Initialize the RAG vector store from persisted data or create new one."""
    global _vector_store
    
    if _vector_store is not None:
        return _vector_store
    
    # Try to load existing vector store
    if os.path.exists(os.path.join(persist_directory, 'vectorstore.pkl')):
        _vector_store = SimpleVectorStore().load(persist_directory)
        return _vector_store
    
    # Create new vector store from documents
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
    
    # Create simulated vector store
    _vector_store = SimpleVectorStore().from_documents(
        documents=chunks,
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
    db_path = "data/energy_data.db"
    session = get_session(db_path)
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Query total energy usage (sum across all devices)
        total_query = session.query(
            func.sum(EnergyUsage.energy_kwh).label("total_usage"),
            func.sum(EnergyUsage.cost_usd).label("total_cost")
        ).filter(
            and_(
                EnergyUsage.timestamp >= start,
                EnergyUsage.timestamp <= end
            )
        )
        
        total_result = total_query.first()
        
        if total_result.total_usage is None:
            return f"No energy usage data found for {start_date} to {end_date}"
        
        # Query usage by device type
        by_device_query = session.query(
            EnergyUsage.device_type,
            func.sum(EnergyUsage.energy_kwh).label("total_kwh"),
            func.sum(EnergyUsage.cost_usd).label("total_cost")
        ).filter(
            and_(
                EnergyUsage.timestamp >= start,
                EnergyUsage.timestamp <= end
            )
        ).group_by(EnergyUsage.device_type)
        
        device_results = by_device_query.all()
        
        # Calculate days in range
        days = (end - start).days + 1
        
        # Format output
        output = f"Energy Usage Summary ({start_date} to {end_date}):\n\n"
        output += f"Total Usage: {total_result.total_usage:.2f} kWh over {days} days\n"
        output += f"Average Daily Usage: {total_result.total_usage/days:.2f} kWh\n"
        output += f"Total Cost: ${total_result.total_cost:.2f}\n"
        output += f"Average Daily Cost: ${total_result.total_cost/days:.2f}\n\n"
        
        output += "Usage Breakdown by Device Type:\n"
        device_data = {}
        for row in device_results:
            device_type = row.device_type
            kwh = row.total_kwh
            cost = row.total_cost
            pct = (kwh / total_result.total_usage * 100) if total_result.total_usage > 0 else 0
            output += f"  {device_type.upper()}: {kwh:.2f} kWh/period ({pct:.1f}%), ${cost:.2f}\n"
            device_data[device_type] = {'kwh': kwh, 'pct': pct}
        
        # Add insights
        output += "\nInsights:\n"
        
        # HVAC insights
        if 'hvac' in device_data:
            hvac_pct = device_data['hvac']['pct']
            if hvac_pct > 50:
                output += f"- HVAC is {hvac_pct:.0f}% of usage - consider optimizing thermostat settings\n"
            elif hvac_pct > 40:
                output += f"- HVAC usage at {hvac_pct:.0f}% is typical for many homes\n"
            else:
                output += f"- HVAC usage at {hvac_pct:.0f}% is relatively low - good job!\n"
        
        # EV insights
        if 'ev_charger' in device_data:
            ev_pct = device_data['ev_charger']['pct']
            if ev_pct > 30:
                output += f"- High EV charging at {ev_pct:.0f}% - ensure charging during off-peak or solar hours\n"
            else:
                output += f"- EV charging at {ev_pct:.0f}% of usage - monitor charging schedules\n"
        
        # Daily usage insights
        avg_daily = total_result.total_usage / days
        if avg_daily > 40:
            output += f"- Average daily usage of {avg_daily:.1f} kWh is above typical (30-35 kWh)\n"
            output += "  Consider energy audit to identify savings opportunities\n"
        elif avg_daily < 25:
            output += f"- Excellent! Average usage of {avg_daily:.1f} kWh is below typical\n"
        
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
    db_path = "data/energy_data.db"
    session = get_session(db_path)
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Query solar generation (use correct column name: generation_kwh)
        query = session.query(
            func.sum(SolarGeneration.generation_kwh).label("total_generation"),
            func.avg(SolarGeneration.generation_kwh).label("avg_generation"),
            func.max(SolarGeneration.generation_kwh).label("max_generation"),
            func.min(SolarGeneration.generation_kwh).label("min_generation"),
            func.count(SolarGeneration.id).label("record_count")
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
            func.sum(EnergyUsage.energy_kwh).label("total_usage")
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
        output += f"Average Daily Generation: {result.total_generation/days:.2f} kWh\n"
        output += f"Peak Hourly Generation: {result.max_generation:.2f} kWh\n"
        output += f"Minimum Hourly Generation: {result.min_generation:.2f} kWh\n\n"
        
        # Calculate weather-based insights from data
        sunny_query = session.query(
            func.avg(SolarGeneration.generation_kwh).label("avg_gen")
        ).filter(
            and_(
                SolarGeneration.timestamp >= start,
                SolarGeneration.timestamp <= end,
                SolarGeneration.weather_condition == 'sunny'
            )
        )
        sunny_result = sunny_query.first()
        
        if sunny_result.avg_gen:
            output += f"Average Generation on Sunny Days: {sunny_result.avg_gen:.2f} kWh/hour\n\n"
        
        # Add insights
        output += "Insights:\n"
        
        # Solar offset calculation
        if usage_result.total_usage and usage_result.total_usage > 0:
            solar_offset = (result.total_generation / usage_result.total_usage) * 100
            output += f"- Solar offsets {solar_offset:.0f}% of your energy usage\n"
            if solar_offset >= 100:
                output += "  Excellent! You're generating more than you use\n"
            elif solar_offset >= 70:
                output += "  Great solar coverage!\n"
            else:
                output += "  Consider expanding solar array or reducing usage\n"
        
        # Daily generation assessment
        avg_daily = result.total_generation / days
        if avg_daily > 35:
            output += f"- Strong daily generation of {avg_daily:.1f} kWh\n"
        elif avg_daily > 20:
            output += f"- Moderate daily generation of {avg_daily:.1f} kWh\n"
        else:
            output += f"- Generation of {avg_daily:.1f} kWh/day - check for shading or maintenance needs\n"
        
        # Peak generation insights
        if result.max_generation > 5:
            output += f"- Peak generation of {result.max_generation:.1f} kWh shows good system capacity\n"
        
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

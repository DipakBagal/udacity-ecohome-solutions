"""Energy data models for EcoHome."""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class EnergyUsage(Base):
    """Energy usage data model."""
    __tablename__ = 'energy_usage'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    device_type = Column(String(50), nullable=False)  # e.g., 'HVAC', 'EV', 'Dishwasher'
    device_name = Column(String(100), nullable=False)
    energy_kwh = Column(Float, nullable=False)  # Energy consumed in kWh
    cost_usd = Column(Float, nullable=False)  # Cost in USD
    price_per_kwh = Column(Float, nullable=False)  # Price per kWh at that time
    
    def __repr__(self):
        return f"<EnergyUsage(device={self.device_name}, energy={self.energy_kwh}kWh, cost=${self.cost_usd})>"


class SolarGeneration(Base):
    """Solar generation data model."""
    __tablename__ = 'solar_generation'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    generation_kwh = Column(Float, nullable=False)  # Solar energy generated in kWh
    weather_condition = Column(String(50))  # e.g., 'Sunny', 'Cloudy', 'Rainy'
    temperature_f = Column(Float)  # Temperature in Fahrenheit
    cloud_cover_percent = Column(Integer)  # Cloud cover percentage
    
    def __repr__(self):
        return f"<SolarGeneration(generation={self.generation_kwh}kWh, weather={self.weather_condition})>"


def get_engine(db_path="data/energy_data.db"):
    """Create and return database engine."""
    return create_engine(f'sqlite:///{db_path}')


def get_session(db_path="data/energy_data.db"):
    """Create and return database session."""
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(db_path="data/energy_data.db"):
    """Initialize database with tables."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

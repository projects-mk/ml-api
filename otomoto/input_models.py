from pydantic import BaseModel

class OtomotoInputData(BaseModel):
    model_name: str
    vehicle_brand: str
    vehicle_model: str
    vehicle_version: str
    vehicle_generation: str
    year_of_production: int
    mileage: float
    engine_capacity: float
    fuel_type: str
    horse_power: float
    transmission_type: str
    drive_type: str
    gas_usage_per_100km: float
    car_body_type: str
    number_of_doors: int
    number_of_seats: float
    color: str
    country_of_origin: str
    new_used: str
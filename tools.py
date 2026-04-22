def get_battery_status():
    return {"battery": "78%"}

def set_temperature(temp):
    return {"message": f"Cabin temperature set to {temp}°C"}

def open_sunroof():
    return {"message": "Sunroof opened"}

def get_vehicle_health():
    return {
        "tire_pressure": "Normal (32 PSI)",
        "brake_pads": "80% Life",
        "next_service": "5,000 km",
        "oil_level": "Optimal"
    }



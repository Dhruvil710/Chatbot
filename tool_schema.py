tools = [
    {
        "type": "function",
        "function": {
            "name": "get_battery_status",
            "description": "Get current battery level of the car",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set the cabin temperature",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "number",
                        "description": "temperature in Celsius"
                    }
                },
                "required": ["temp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_sunroof",
            "description": "Open the car's sunroof",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_health",
            "description": "Get a full diagnostic report",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


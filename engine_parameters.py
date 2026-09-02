"""
Rotax 914 Reference Engine Parameters
Digital Twin Project
"""

ENGINE = {

    # Basic engine configuration
    "name": "Rotax 914 UL/F",
    "engine_type": "4-stroke piston",
    "cylinders": 4,
    "turbocharged": True,

    # Engine geometry
    "displacement_cm3": 1211.2,
    "bore_mm": 79.5,
    "stroke_mm": 61.0,
    "compression_ratio": 9.0,

    # Performance
    "max_power_kw": 84.8,
    "max_power_hp": 115,
    "max_rpm": 5800,
    "max_torque_nm": 144,
    "max_torque_rpm": 4900,

    # Propeller gearbox
    "propeller_reduction_ratio": 2.43,

    # Fuel and ignition
    "fuel_system": "2 carburetors",
    "ignition": "Dual electronic ignition",

    # Lubrication
    "lubrication_type": "Dry sump forced lubrication",

    # Service life
    "tbo_hours": 2000,

    # Weight
    "engine_weight_kg": 64.0,
}
import math


# Standard atmosphere constants
SEA_LEVEL_TEMPERATURE_K = 288.15
SEA_LEVEL_PRESSURE_PA = 101325.0

TEMPERATURE_LAPSE_RATE = 0.0065
GRAVITY = 9.80665
AIR_GAS_CONSTANT = 287.05


def calculate_atmosphere(altitude_m):
    """
    Estimate atmospheric conditions at a given altitude.

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level in metres.

    Returns
    -------
    dict
        Atmospheric temperature, pressure and air density.
    """

    altitude_m = max(0, altitude_m)

    temperature_k = (
        SEA_LEVEL_TEMPERATURE_K
        - TEMPERATURE_LAPSE_RATE * altitude_m
    )

    pressure_pa = SEA_LEVEL_PRESSURE_PA * (
        temperature_k / SEA_LEVEL_TEMPERATURE_K
    ) ** (
        GRAVITY /
        (AIR_GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
    )

    density_kg_m3 = pressure_pa / (
        AIR_GAS_CONSTANT * temperature_k
    )

    return {
        "temperature_k": temperature_k,
        "pressure_pa": pressure_pa,
        "density_kg_m3": density_kg_m3
    }

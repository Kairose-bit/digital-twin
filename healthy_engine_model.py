from engine_parameters import ENGINE
from atmosphere_model import calculate_atmosphere


def calculate_engine_state(
    throttle,
    altitude_m,
    ambient_temp_c
):
    """
    Calculate the estimated healthy engine state.

    Parameters
    ----------
    throttle : float
        Throttle position from 0.0 to 1.0.

    altitude_m : float
        Altitude above sea level in metres.

    ambient_temp_c : float
        Ambient temperature in Celsius.

    Returns
    -------
    dict
        Estimated engine state.
    """

    # -----------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------

    throttle = max(0.0, min(1.0, throttle))
    altitude_m = max(0.0, altitude_m)

    # -----------------------------------------
    # ATMOSPHERE
    # -----------------------------------------

    atmosphere = calculate_atmosphere(altitude_m)

    air_density = atmosphere["density_kg_m3"]

    # -----------------------------------------
    # ALTITUDE EFFECT
    # -----------------------------------------

    sea_level_density = 1.225

    density_ratio = air_density / sea_level_density

    # The turbocharged engine can compensate
    # for part of the altitude-related loss.
    # This is a simplified prototype assumption.

    turbo_compensation = min(
        1.0,
        0.75 + 0.25 * density_ratio
    )

    # -----------------------------------------
    # ENGINE LOAD
    # -----------------------------------------

    engine_load = throttle * turbo_compensation

    # -----------------------------------------
    # RPM
    # -----------------------------------------

    idle_rpm = 2000
    max_rpm = ENGINE["max_rpm"]

    rpm = (
        idle_rpm
        + engine_load * (max_rpm - idle_rpm)
    )

    # -----------------------------------------
    # EGT
    # -----------------------------------------

    base_egt = 450
    max_egt = 700

    egt_c = (
        base_egt
        + engine_load * (max_egt - base_egt)
    )

    # -----------------------------------------
    # CHT
    # -----------------------------------------

    base_cht = 90
    max_cht = 180

    cht_c = (
        base_cht
        + engine_load * (max_cht - base_cht)
    )

    # Ambient temperature effect
    cht_c += (ambient_temp_c - 20) * 0.5

    # -----------------------------------------
    # OIL PRESSURE
    # -----------------------------------------

    oil_pressure_psi = (
        30
        + engine_load * 30
    )

    # -----------------------------------------
    # OIL TEMPERATURE
    # -----------------------------------------

    oil_temperature_c = (
        70
        + engine_load * 40
        + (ambient_temp_c - 20) * 0.3
    )

    # -----------------------------------------
    # FUEL FLOW
    # -----------------------------------------

    max_fuel_flow = 25

    fuel_flow = engine_load * max_fuel_flow

    # -----------------------------------------
    # RETURN ENGINE STATE
    # -----------------------------------------

    return {
        "altitude_m": round(altitude_m, 2),
        "ambient_temp_c": round(ambient_temp_c, 2),

        "air_density_kg_m3": round(
            air_density, 4
        ),

        "throttle": round(throttle, 3),

        "rpm": round(rpm, 2),

        "egt_c": round(egt_c, 2),

        "cht_c": round(cht_c, 2),

        "oil_pressure_psi": round(
            oil_pressure_psi, 2
        ),

        "oil_temperature_c": round(
            oil_temperature_c, 2
        ),

        "fuel_flow": round(
            fuel_flow, 2
        ),

        "engine_load": round(
            engine_load, 3
        ),

        "health_state": "NORMAL"
    }
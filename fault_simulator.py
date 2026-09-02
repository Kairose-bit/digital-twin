import random


def apply_fault(engine_state, fault_type):
    """
    Apply a simplified fault/degradation effect
    to a healthy Digital Twin engine state.

    These are prototype synthetic relationships,
    not certified engine failure limits.
    """

    state = engine_state.copy()

    # -----------------------------------------
    # NORMAL
    # -----------------------------------------

    if fault_type == "NORMAL":
        state["fault_severity"] = 0.0

    # -----------------------------------------
    # MISFIRE
    # -----------------------------------------

    elif fault_type == "MISFIRE":

        severity = random.uniform(0.3, 1.0)

        state["rpm"] *= (
            1 - 0.04 * severity
        )

        state["egt_c"] *= (
            1 - 0.08 * severity
        )

        state["cht_c"] *= (
            1 - 0.03 * severity
        )

        state["vibration_rms_g"] *= (
            1 + 0.60 * severity
        )

        state["fault_severity"] = severity

    # -----------------------------------------
    # INJECTOR ABNORMALITY
    # -----------------------------------------

    elif fault_type == "INJECTOR_ABNORMALITY":

        severity = random.uniform(0.3, 1.0)

        state["fuel_flow"] *= (
            1 + 0.20 * severity
        )

        state["egt_c"] *= (
            1 + 0.10 * severity
        )

        state["rpm"] *= (
            1 - 0.03 * severity
        )

        state["vibration_rms_g"] *= (
            1 + 0.35 * severity
        )

        state["fault_severity"] = severity

    # -----------------------------------------
    # LUBRICATION ISSUE
    # -----------------------------------------

    elif fault_type == "LUBRICATION_ISSUE":

        severity = random.uniform(0.3, 1.0)

        state["oil_pressure_psi"] *= (
            1 - 0.35 * severity
        )

        state["oil_temperature_c"] *= (
            1 + 0.20 * severity
        )

        state["cht_c"] *= (
            1 + 0.08 * severity
        )

        state["vibration_rms_g"] *= (
            1 + 0.30 * severity
        )

        state["fault_severity"] = severity

    # -----------------------------------------
    # OVERHEATING
    # -----------------------------------------

    elif fault_type == "OVERHEATING":

        severity = random.uniform(0.3, 1.0)

        state["cht_c"] *= (
            1 + 0.25 * severity
        )

        state["egt_c"] *= (
            1 + 0.15 * severity
        )

        state["oil_temperature_c"] *= (
            1 + 0.20 * severity
        )

        state["fault_severity"] = severity

    # -----------------------------------------
    # COMBUSTION INSTABILITY
    # -----------------------------------------

    elif fault_type == "COMBUSTION_INSTABILITY":

        severity = random.uniform(0.3, 1.0)

        state["rpm"] *= (
            1 + random.uniform(-0.04, 0.04) * severity
        )

        state["egt_c"] *= (
            1 + random.uniform(-0.10, 0.10) * severity
        )

        state["cht_c"] *= (
            1 + random.uniform(-0.05, 0.05) * severity
        )

        state["vibration_rms_g"] *= (
            1 + 0.50 * severity
        )

        state["fault_severity"] = severity

    # -----------------------------------------
    # ABNORMAL VIBRATION
    # -----------------------------------------

    elif fault_type == "ABNORMAL_VIBRATION":

        severity = random.uniform(0.3, 1.0)

        state["vibration_rms_g"] *= (
            1 + 1.00 * severity
        )

        state["rpm"] *= (
            1 + random.uniform(-0.02, 0.02)
        )

        state["fault_severity"] = severity

    # -----------------------------------------
    # SENSOR DRIFT
    # -----------------------------------------

    elif fault_type == "SENSOR_DRIFT":

        severity = random.uniform(0.3, 1.0)

        # The physical engine remains healthy.
        # Only sensor readings drift.

        state["egt_c"] *= (
            1 + 0.08 * severity
        )

        state["cht_c"] *= (
            1 - 0.06 * severity
        )

        state["oil_pressure_psi"] *= (
            1 + 0.10 * severity
        )

        state["fault_severity"] = severity

    else:
        raise ValueError(
            f"Unknown fault type: {fault_type}"
        )

    state["health_state"] = fault_type

    return state
import csv
import random

from healthy_engine_model import calculate_engine_state
from fault_simulator import apply_fault


OUTPUT_FILE = "engine_dataset_v1.csv"

SAMPLES_PER_CLASS = 1000


FAULT_TYPES = [
    "NORMAL",
    "MISFIRE",
    "INJECTOR_ABNORMALITY",
    "LUBRICATION_ISSUE",
    "OVERHEATING",
    "COMBUSTION_INSTABILITY",
    "ABNORMAL_VIBRATION",
    "SENSOR_DRIFT"
]


def generate_dataset():

    rows = []

    sample_number = 0

    for fault_type in FAULT_TYPES:

        for _ in range(SAMPLES_PER_CLASS):

            # -----------------------------------------
            # OPERATING CONDITIONS
            # -----------------------------------------

            throttle = random.uniform(
                0.2,
                1.0
            )

            altitude = random.uniform(
                0,
                5000
            )

            ambient_temperature = random.uniform(
                5,
                40
            )

            # -----------------------------------------
            # HEALTHY DIGITAL TWIN
            # -----------------------------------------

            engine_state = calculate_engine_state(
                throttle=throttle,
                altitude_m=altitude,
                ambient_temp_c=ambient_temperature
            )

            # -----------------------------------------
            # ADD SENSOR PARAMETERS
            # -----------------------------------------

            engine_state["vibration_rms_g"] = (
                0.02
                + 0.03 * engine_state["engine_load"]
                + random.gauss(0, 0.003)
            )

            engine_state["injection_timing_deg"] = (
                25
                - 3 * engine_state["engine_load"]
                + random.gauss(0, 0.3)
            )

            engine_state["battery_voltage_v"] = (
                14.0
                + random.gauss(0, 0.1)
            )

            engine_state["alternator_current_a"] = (
                10
                + 15 * engine_state["engine_load"]
                + random.gauss(0, 0.5)
            )

            # -----------------------------------------
            # APPLY FAULT
            # -----------------------------------------

            engine_state = apply_fault(
                engine_state,
                fault_type
            )

            # -----------------------------------------
            # ADD DATASET METADATA
            # -----------------------------------------

            engine_state["sample"] = sample_number

            engine_state["fault"] = fault_type

            rows.append(engine_state)

            sample_number += 1

    # ---------------------------------------------
    # SAVE DATASET
    # ---------------------------------------------

    fieldnames = [
        "sample",

        "altitude_m",
        "ambient_temp_c",
        "air_density_kg_m3",
        "throttle",

        "rpm",
        "egt_c",
        "cht_c",

        "oil_pressure_psi",
        "oil_temperature_c",

        "fuel_flow",
        "engine_load",

        "vibration_rms_g",
        "injection_timing_deg",

        "battery_voltage_v",
        "alternator_current_a",

        "fault_severity",
        "fault",
        "health_state"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("========================================")
    print(" DIGITAL TWIN DATASET GENERATED")
    print("========================================")
    print(f"Total samples: {len(rows)}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Fault classes: {len(FAULT_TYPES)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    generate_dataset()
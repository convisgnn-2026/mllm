import json

MODELS = ["flux", "sd3"]
gpu_names = [
    "L40",
    "H100_HBM3",
]


def merge(LOG_DIR):
    with open(f"{LOG_DIR}/e2e_scaling_efficiency_1.json", "r") as f:
        data_1 = json.load(f)

    with open(f"{LOG_DIR}/e2e_scaling_efficiency_2.json", "r") as f:
        data_2 = json.load(f)

    with open(f"{LOG_DIR}/e2e_scaling_efficiency_4.json", "r") as f:
        data_4 = json.load(f)

    try:
        with open(f"{LOG_DIR}/e2e_scaling_efficiency_8.json", "r") as f:
            data_8 = json.load(f)
    except FileNotFoundError:
        data_8 = []

    data = data_1 + data_2 + data_4 + data_8

    with open(f"{LOG_DIR}/e2e_scaling_efficiency.json", "w") as f:
        json.dump(data, f, indent=2)


for model in MODELS:
    for gpu_name in gpu_names:
        LOG_DIR = f"log/benchmark/scaling_efficiency/{model}/{gpu_name}"
        try:
            merge(LOG_DIR)
            print(f"Merged data for {LOG_DIR}")
        except FileNotFoundError as e:
            print(f"No data found for {LOG_DIR}: {e}")
            continue

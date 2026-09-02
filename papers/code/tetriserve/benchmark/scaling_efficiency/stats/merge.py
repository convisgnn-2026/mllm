import json


LOG_DIR = "log/benchmark/scaling_efficiency/sd3/stats"


with open(f"{LOG_DIR}/e2e_scaling_efficiency_stats_1.json", "r") as f:
    data_1 = json.load(f)

with open(f"{LOG_DIR}/e2e_scaling_efficiency_stats_2.json", "r") as f:
    data_2 = json.load(f)

with open(f"{LOG_DIR}/e2e_scaling_efficiency_stats_4.json", "r") as f:
    data_4 = json.load(f)

# with open(f"{LOG_DIR}/e2e_scaling_efficiency_stats_8.json", "r") as f:
#     data_8 = json.load(f)
data_8 = []

data = data_1 + data_2 + data_4 + data_8

with open(f"{LOG_DIR}/e2e_scaling_efficiency_stats.json", "w") as f:
    json.dump(data, f, indent=2)

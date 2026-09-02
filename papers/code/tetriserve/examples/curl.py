import requests

response = requests.post(
    "http://localhost:1037/v1/generate",
    json={
        "prompt": ["A beautiful sunset over mountains"] * 4,
        "batch_size": 4,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 20,
        "latency_threshold": 16,
    },
)

# If output_type is "pil", you'll get base64 encoded images
images = response.json()["images"]

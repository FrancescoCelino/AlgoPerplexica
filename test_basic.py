import requests
from dotenv import load_dotenv
import os
load_dotenv()

api_port = os.getenv("API_PORT", 8000)

def test_api(): 
    url = f"http://localhost:{api_port}/enrich"
    data = {
        "av_brain_prompt": "What are the lastest LLMs released?",
        "av_brain_context": "--- Company context --- \n\n Here we have some business context about the company and its products."
    }

    response = requests.post(url, json=data)
    print("Response status code:", response.status_code)
    print(f"Response JSON: {response.json()}")

if __name__ == "__main__":
    test_api()
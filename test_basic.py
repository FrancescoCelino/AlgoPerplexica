import requests
from dotenv import load_dotenv
import os
load_dotenv()

api_port = os.getenv("API_PORT", 8000)

def test_api(): 
    url = f"http://localhost:{api_port}/enrich"
    data = {
        "av_brain_prompt": "What are some resources I could use to invest in some skillas that are marketable for my job?",
        "av_brain_context": "--- Company context --- \n\n I'm a freelance data scientist and AI engineer, based in Italy and remote working for an american company. I have a background in mathematics and engineering but I still need to work on my coding skills and soft skills."
    }

    response = requests.post(url, json=data)
    print("Response status code:", response.status_code)
    print(f"Response JSON: {response.json()}")

if __name__ == "__main__":
    test_api()
import requests
import os
from dotenv import load_dotenv

load_dotenv()

PERPLEXICA_URL = os.getenv("PERPLEXICA_URL", "http://localhost:3000")

def process_av_brain(av_brain_prompt: str, av_brain_context: str) -> str: 
    # TODO: improve query
    query: str = f"{av_brain_prompt}\n\n{av_brain_context}"
    try:
        perplexica_output = search_perplexica(query)
    except Exception as e:
        print(f"Error during Perplexica search: {e}")
        print("Returning context as fallback.")
        return av_brain_context
    
    enriched_context: str = f"{av_brain_context}\n\n ---Additional Information from Web Search ---\n{perplexica_output['message']}"
    return enriched_context



def search_perplexica(question: str) -> dict: 

    url = f"{PERPLEXICA_URL}/api/search"

    data = {
        "chatModel": {
            "provider": "ollama",
            "name": "llama3:latest"
        },
        "embeddingModel": {
            "provider": "ollama", 
            "name": "nomic-embed-text:latest"
        },
        "query": question,
        "focusMode": "webSearch"}
    
    response = requests.post(url, json=data)
    result = response.json()

    return result

def get_available_models() -> None:
    url = f"{PERPLEXICA_URL}/api/models"

    response = requests.get(url)
    models = response.json()
    print("DEBUG: available models")
    print(models)
    print("available chat models")
    for provider, provider_models in models["chatModelProviders"].items():
        print(f" {provider}:")
        for model_name, model_info in provider_models.items():
            print(f"  {model_name} - {model_info['displayName']}")
    print("\navailable embedding models")
    for provider, provider_models in models["embeddingModelProviders"].items():
        print(f" {provider}:")
        for model_name, model_info in provider_models.items():
            print(f"  {model_name} - {model_info['displayName']}")
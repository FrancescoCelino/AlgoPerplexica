import requests
import json

def search_perplexica(question: str): 
    url = "http://localhost:3000/api/search"

    data = {
        "query": question,
        "focusMode": "webSearch"}
    
    response = requests.post(url, json=data)
    result = response.json()

    return result

def search_perplexica_with_llama(question: str): 
    url = "http://localhost:3000/api/search"

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
    url = "http://localhost:3000/api/models"

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

get_available_models()

# perplexica_result = search_perplexica("Current price of Tesla Stock")
perplexica_result = search_perplexica_with_llama("Current price of Tesla Stock")

print(perplexica_result["message"])
# print("printing the full result:")
# print(perplexica_result)
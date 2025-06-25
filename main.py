import requests
import os
from dotenv import load_dotenv

load_dotenv()

PERPLEXICA_URL = os.getenv("PERPLEXICA_URL", "http://localhost:3000")

def enrich_context_from_queries(queries: list[str], av_brain_context) -> str: 
    # TODO: improve query
    extra_context: str = ""
    for query in queries:
        query_result: dict = search_perplexica(query)
        if query_result.get("message"):
            extra_context = extra_context + f"\n\n ---Results for query: {query} ---\n {query_result['message']}"
        else: 
            print(f"WARNING: No results found for query: {query}")
    if not extra_context:
        print("No results found for any of the queries. Please check your queries or the Perplexica service.")
        return av_brain_context
    
    enriched_context: str = f"{av_brain_context}\n\n ---Additional Information from Web Search ---\n{extra_context}"
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

def extract_queries_from_av_brain(av_brain_prompt: str, av_brain_context: str, max_number_of_queries: int = 3) -> list[str]: 
    queries = []
    # TODO: this part is problematic because it includes the context in the query. We need to generate queries that are more focused.
    # For now, we will just use the prompt and context as queries. However, doing so Perplexica will also include the context in the response. 
    #queries.append(av_brain_prompt + " " + av_brain_context)

    queries = ["latest LLMs released", "cutest cat breeds", "latest videogames released"] # example

    return queries
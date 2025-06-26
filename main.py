import requests
import os
from dotenv import load_dotenv
import json
import re
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


def extract_queries_from_av_brain(av_brain_prompt: str, av_brain_context: str, max_number_of_queries: int = 2) -> list[str]: 
    queries = []
    
    llm_prompt = f"""You are an expert at generating web search queries. Given a user's question and company context, generate {max_number_of_queries} focused search queries.

User Question: {av_brain_prompt}
Company Context: {av_brain_context}

Return your response as a valid JSON array of strings. Example format:
["query 1", "query 2", "query 3"]

Do not include any other text, just the JSON array:"""
    LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434")
    try: 
        response = requests.post(
            f"{LLAMA_API_URL}/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": llm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  
                    "top_p": 0.9
                }
            }        )
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            print("Response was good, checking generated text...")
            try: 
                queries = json.loads(generated_text)
                print(f"DEBUG: Parsed JSON successfully: {queries}")
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    clean_queries = [q.strip() for q in queries if q.strip() and len(q.strip()) > 3]
                    clean_queries = clean_queries[:max_number_of_queries]
                    print(f"DEBUG: Formatting is correct, Parsed {len(clean_queries)} queries from LLM response: {clean_queries}")
                    if clean_queries:
                        print(f"DEBUG: Successfully parsed JSON queries: {clean_queries}")
                        return clean_queries
            except json.JSONDecodeError:
                print("DEBUG: Failed to parse JSON from generated text, falling back to text parsing")
                return parse_queries_from_text(generated_text, av_brain_prompt, max_number_of_queries)
        print("LLM response was not successful, returning AVBrain prompt.")
        return [av_brain_prompt]
    
    except Exception as e:
        print(f"DEBUG: Error during LLM query generation: {e}")
        # Fallback to a simple extraction if LLM fails
        return [av_brain_prompt]

    # queries = ["latest LLMs released", "cutest cat breeds", "latest videogames released"] # example

    return queries

def parse_queries_from_text(text: str, fallback_query: str, max_queries: int = 2) -> list[str]: 
    queries = []
    print("WARNING: we are parsing queries from text")
    # Method 1: Try to extract JSON from the text
    json_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if json_match:
        try:
            # Try to parse the bracketed content as JSON
            json_content = '[' + json_match.group(1) + ']'
            parsed = json.loads(json_content)
            if isinstance(parsed, list):
                queries = [str(q).strip().strip('"\'') for q in parsed]
        except:
            pass
    
    # Method 2: Line-by-line parsing with cleaning
    if not queries:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and common intro/outro phrases
            skip_phrases = [
                'here are', 'search queries', 'queries:', 'these queries',
                'should help', 'you can use', 'try searching', ''
            ]
            
            if any(phrase in line.lower() for phrase in skip_phrases):
                continue
                
            # Remove common prefixes
            prefixes = [r'^\d+\.?\s*', r'^-\s*', r'^\*\s*', r'^•\s*', r'^["\']', r'["\']$']
            for prefix in prefixes:
                line = re.sub(prefix, '', line).strip()
            
            # Remove quotes at start/end
            line = line.strip('\'"')
            
            if len(line) > 3 and len(line) < 100:  # Reasonable query length
                queries.append(line)
    
    # Method 3: Extract quoted strings
    if not queries:
        quoted_strings = re.findall(r'"([^"]+)"', text)
        queries = [q.strip() for q in quoted_strings if len(q.strip()) > 3]
    
    # Method 4: Split by common delimiters and clean
    if not queries:
        # Try splitting by newlines, then by commas
        for delimiter in ['\n', ',']:
            parts = text.split(delimiter)
            potential_queries = []
            for part in parts:
                cleaned = re.sub(r'^\d+\.?\s*|-\s*|\*\s*', '', part.strip())
                if 3 < len(cleaned) < 100:
                    potential_queries.append(cleaned)
            if potential_queries:
                queries = potential_queries
                break
    
    # Filter and limit
    queries = [q for q in queries if q and len(q) > 3][:max_queries]
    
    if not queries:
        print("WARNING: Could not parse any queries from LLM output, using fallback")
        return [fallback_query]
    
    print(f"DEBUG: Parsed {len(queries)} queries from text: {queries}")
    return queries

def extract_queries_simple_fallback(av_brain_prompt: str, av_brain_context: str, max_number_of_queries = 2) -> list[str]:
    """
    Simpler approach: Just use the prompt and let Perplexica handle it
    This is your current working approach as a reliable fallback
    """
    # Extract key terms from the prompt
    # This is more reliable than depending on LLM formatting
    
    base_query = av_brain_prompt.strip()
    
    # You could add simple variations:
    queries = [base_query]
    
    # TODO: add related queries to the list using some simple heuristics
    
    return queries[:max_number_of_queries]
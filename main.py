import requests
import os
from dotenv import load_dotenv
import json
import re
from typing import Optional, Dict, Tuple
load_dotenv()

PERPLEXICA_URL = os.getenv("PERPLEXICA_URL", "http://localhost:3000")

_model_cache = {
    "chat_models": {},      
    "embedding_models": {}, 
    "last_updated": None,
    "perplexica_available": False
}

def get_perplexica_models() -> Dict:
    """
    Fetch available models from Perplexica API
    Returns the raw response from /api/models
    """
    try:
        url = f"{PERPLEXICA_URL}/api/models"
        print(f"DEBUG: Attempting to connect to Perplexica at {url}")
        response = requests.get(url, timeout=5)  # Shorter timeout
        
        if response.status_code == 200:
            print("DEBUG: Successfully connected to Perplexica API")
            return response.json()
        else:
            print(f"ERROR: Perplexica models API returned {response.status_code}")
            return {}
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Perplexica - is it running?")
        print("       Start Perplexica with: docker-compose up -d")
        return {}
    except requests.exceptions.Timeout:
        print("ERROR: Perplexica API timeout")
        return {}
    except Exception as e:
        print(f"ERROR: Failed to fetch Perplexica models: {e}")
        return {}


def get_fallback_mapping() -> Dict[str, Dict[str, str]]:
    """
    Fallback model mapping when Perplexica API is unavailable
    """
    return {
        "chat_models": {
            "llama3:latest": "ollama",
            "llama3:8b": "ollama",
            "llama3:70b": "ollama", 
            "codellama:latest": "ollama",
            "mistral:latest": "ollama",
            "gpt-3.5-turbo": "openai", 
            "gpt-4": "openai",
            "gpt-4-turbo": "openai",
            "claude-3-haiku-20240307": "anthropic",
            "claude-3-sonnet-20240229": "anthropic",
            "claude-3-opus-20240229": "anthropic"
        },
        "embedding_models": {
            "nomic-embed-text:latest": "ollama",
            "mxbai-embed-large:latest": "ollama",
            "text-embedding-3-small": "openai",
            "text-embedding-3-large": "openai",
            "text-embedding-ada-002": "openai"
        }
    }


def build_model_to_provider_mapping() -> Dict[str, Dict[str, str]]:
    """
    Build mapping from model names to their providers
    
    Returns:
    {
        "chat_models": {"llama3:latest": "ollama", "gpt-4": "openai", ...},
        "embedding_models": {"nomic-embed-text:latest": "ollama", ...}
    }
    """
    models_data = get_perplexica_models()
    
    if not models_data:
        print("WARNING: Using fallback model mapping (Perplexica unavailable)")
        _model_cache["perplexica_available"] = False
        return get_fallback_mapping()
    
    _model_cache["perplexica_available"] = True
    mapping = {"chat_models": {}, "embedding_models": {}}
    
    try:
        # Build chat models mapping
        if "chatModelProviders" in models_data:
            for provider, provider_models in models_data["chatModelProviders"].items():
                # ✅ Fix: Handle both list and dict formats
                if isinstance(provider_models, list):
                    # Format: [{"name": "llama3:latest", "displayName": "..."}]
                    for model_info in provider_models:
                        if isinstance(model_info, dict) and "name" in model_info:
                            model_name = model_info["name"]
                            mapping["chat_models"][model_name] = provider
                elif isinstance(provider_models, dict):
                    # Format: {"llama3:latest": {"displayName": "..."}}
                    for model_name, model_info in provider_models.items():
                        mapping["chat_models"][model_name] = provider
        
        # Build embedding models mapping  
        if "embeddingModelProviders" in models_data:
            for provider, provider_models in models_data["embeddingModelProviders"].items():
                # ✅ Fix: Handle both list and dict formats
                if isinstance(provider_models, list):
                    for model_info in provider_models:
                        if isinstance(model_info, dict) and "name" in model_info:
                            model_name = model_info["name"]
                            mapping["embedding_models"][model_name] = provider
                elif isinstance(provider_models, dict):
                    for model_name, model_info in provider_models.items():
                        mapping["embedding_models"][model_name] = provider
        
        print(f"DEBUG: Successfully built mapping for {len(mapping['chat_models'])} chat models and {len(mapping['embedding_models'])} embedding models")
        return mapping
        
    except Exception as e:
        print(f"ERROR: Failed to parse Perplexica models data: {e}")
        print("WARNING: Falling back to static model mapping")
        _model_cache["perplexica_available"] = False
        return get_fallback_mapping()

def enrich_context_from_queries(queries: list[str], av_brain_context, enriching_model: Optional[str] = None, embedding_model: Optional[str] = None) -> str: 
    
    if enriching_model is None: 
        enriching_model = os.getenv("DEFAULT_CONTEXT_ENRICHMENT_MODEL", "llama3:latest")

    if embedding_model is None:
        embedding_model = os.getenv("PERPLEXICA_EMBEDDING_MODEL", "nomic-embed-text:latest")

    print(f"DEBUG: Using model for context enrichment: {enriching_model}")
    print(f"DEBUG: Using embedding model for Perplexica search: {embedding_model}")
    extra_context: str = ""
    for query in queries:
        print(f"DEBUG: Searching Perplexica for: '{query}' using {enriching_model}")
        
        # Use specified provider for Perplexica search
        query_result: dict = search_perplexica(query, enriching_model=enriching_model, embedding_model=embedding_model)
        
        if query_result.get("message"):
            message = query_result["message"]
            cleaned_message = message.replace(av_brain_context, "").strip()
            cleaned_message = cleaned_message.replace("--- Company context ---", "").strip()
            
            if cleaned_message:
                extra_context += f"\n\n ---Results for query: {query} ---\n {cleaned_message}"
        else: 
            print(f"WARNING: No results found for query: {query}")
    
    if not extra_context:
        print("No results found for any of the queries.")
        return av_brain_context
    
    enriched_context: str = f"{av_brain_context}\n\n ---Additional Information from Web Search ---\n{extra_context}"
    return enriched_context



def search_perplexica(question: str, enriching_model: str, embedding_model: str) -> dict: 
    
    url = f"{PERPLEXICA_URL}/api/search"
    
    # Use dynamic provider detection
    chat_provider = get_model_provider(enriching_model, "chat")
    embedding_provider = get_model_provider(embedding_model, "embedding")
    
    data = {
        "chatModel": {
            "provider": chat_provider,
            "name": enriching_model
        },
        "embeddingModel": {
            "provider": embedding_provider, 
            "name": embedding_model
        },
        "query": question,
        "focusMode": "webSearch"
    }
    
    print(f"DEBUG: Perplexica using chat model: {chat_provider}/{enriching_model}")
    print(f"DEBUG: Perplexica using embedding model: {embedding_provider}/{embedding_model}")
    
    try:
        response = requests.post(url, json=data, timeout=30)
        result = response.json()
        return result
    except Exception as e:
        print(f"ERROR: Perplexica search failed: {e}")
        return {"message": ""}

def get_model_provider(model_name: str, model_type: str = "chat") -> str:
    """
    Get the provider for a specific model name
    
    Args:
        model_name: Name of the model (e.g., "llama3:latest", "gpt-4")
        model_type: "chat" or "embedding"
    
    Returns:
        Provider name (e.g., "ollama", "openai", "anthropic")
    """
    global _model_cache
    
    # Refresh cache if empty
    if not _model_cache["chat_models"] and not _model_cache["embedding_models"]:
        print("DEBUG: Building model-to-provider mapping cache...")
        mapping = build_model_to_provider_mapping()
        _model_cache["chat_models"] = mapping["chat_models"]
        _model_cache["embedding_models"] = mapping["embedding_models"]
    
    # Look up the model
    cache_key = "chat_models" if model_type == "chat" else "embedding_models"
    provider = _model_cache[cache_key].get(model_name)
    
    if provider:
        print(f"DEBUG: Model '{model_name}' ({model_type}) -> Provider '{provider}'")
        return provider
    else:
        print(f"WARNING: Model '{model_name}' not found in {model_type} models, defaulting to 'ollama'")
        return "ollama"  # Safe default

def check_perplexica_status() -> bool:
    """
    Check if Perplexica is running and accessible
    """
    try:
        response = requests.get(f"{PERPLEXICA_URL}/api/models", timeout=3)
        if response.status_code == 200:
            print("✅ Perplexica is running and accessible")
            return True
    except:
        pass
    
    print("❌ Perplexica is not accessible")
    print(f"   URL: {PERPLEXICA_URL}")
    print("   💡 To start Perplexica:")
    print("   - Docker: docker-compose up -d")
    print("   - Or check if it's running on a different port")
    return False


def refresh_model_cache():
    """
    Force refresh the model cache (useful if new models are added to Perplexica)
    """
    global _model_cache
    print("DEBUG: Refreshing model cache...")
    
    mapping = build_model_to_provider_mapping()
    _model_cache["chat_models"] = mapping["chat_models"]
    _model_cache["embedding_models"] = mapping["embedding_models"]
    _model_cache["last_updated"] = requests.get(f"{PERPLEXICA_URL}/api/models").headers.get("date")


def get_available_models_structured() -> Dict:
    """
    Get available models in a structured format
    Works with both live Perplexica data and fallback
    """
    if not _model_cache["chat_models"]:
        # Populate cache first
        get_model_provider("dummy", "chat")  # This will populate cache
    
    structured = {"chat_models": {}, "embedding_models": {}}
    
    # Group models by provider
    for model_name, provider in _model_cache["chat_models"].items():
        if provider not in structured["chat_models"]:
            structured["chat_models"][provider] = []
        
        structured["chat_models"][provider].append({
            "name": model_name,
            "display_name": model_name.replace(":", " ").title()  # Simple display name
        })
    
    for model_name, provider in _model_cache["embedding_models"].items():
        if provider not in structured["embedding_models"]:
            structured["embedding_models"][provider] = []
        
        structured["embedding_models"][provider].append({
            "name": model_name,
            "display_name": model_name.replace(":", " ").title()
        })
    
    return structured


def print_available_models():
    """
    Pretty print available models with status info
    """
    print("=== Available Models ===")
    
    if not _model_cache["perplexica_available"]:
        print("⚠️  Using fallback model list (Perplexica not accessible)")
    else:
        print("✅ Using live data from Perplexica")
    
    models = get_available_models_structured()
    
    print("\n📝 Chat Models:")
    if not models["chat_models"]:
        print("   No chat models available")
    else:
        for provider, provider_models in models["chat_models"].items():
            print(f"  🔧 {provider}:")
            for model in provider_models:
                print(f"    • {model['name']}")
    
    print("\n🔍 Embedding Models:")
    if not models["embedding_models"]:
        print("   No embedding models available")  
    else:
        for provider, provider_models in models["embedding_models"].items():
            print(f"  🔧 {provider}:")
            for model in provider_models:
                print(f"    • {model['name']}")


def extract_queries_from_av_brain(av_brain_prompt: str, av_brain_context: str, max_number_of_queries: int = 2, model: str = None) -> list[str]: 
    
    if model is None:
        model = os.getenv("DEFAULT_QUERY_GENERATION_MODEL", "llama3:latest")
    
    print(f"DEBUG: Using model for query generation: {model}")
    
    # Determine provider from model name
    provider = get_model_provider(model, "chat")
    
    if provider == "ollama":
        return generate_queries_ollama(av_brain_prompt, av_brain_context, max_number_of_queries, model)
    elif provider == "openai":
        return generate_queries_openai(av_brain_prompt, av_brain_context, max_number_of_queries, model)
    elif provider == "anthropic": # TODO: implement this, anthropic is not yet supported
        return generate_queries_anthropic(av_brain_prompt, av_brain_context, max_number_of_queries, model)
    else:
        print(f"WARNING: Unknown provider '{provider}' for model {model}, using ollama")
        return generate_queries_ollama(av_brain_prompt, av_brain_context, max_number_of_queries, "llama3:latest")

def validate_model_exists(model_name: str, model_type: str = "chat") -> bool:
    """
    Check if a model exists in Perplexica
    
    Args:
        model_name: Name of the model to validate
        model_type: "chat" or "embedding"
    
    Returns:
        True if model exists, False otherwise
    """
    cache_key = "chat_models" if model_type == "chat" else "embedding_models"
    
    # Ensure cache is populated
    if not _model_cache[cache_key]:
        get_model_provider(model_name, model_type)  # This will populate cache
    
    return model_name in _model_cache[cache_key]


def generate_queries_ollama(prompt: str, context: str, max_queries: int, model: str = "llama3:latest") -> list[str]:
    """Generate queries using Ollama"""
    llm_prompt = f"""You are an expert at generating web search queries. Given a user's question and company context, generate {max_queries} focused search queries.

User Question: {prompt}
Company Context: {context}

Return your response as a valid JSON array of strings. Example format:
["query 1", "query 2", "query 3"]

Do not include any other text, just the JSON array:"""

    LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434")
    
    try:
        response = requests.post(
            f"{LLAMA_API_URL}/api/generate",
            json={
                "model": model,
                "prompt": llm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            try:
                queries = json.loads(generated_text)
                if isinstance(queries, list):
                    clean_queries = [q.strip() for q in queries if q.strip()][:max_queries]
                    if clean_queries:
                        print(f"DEBUG: Ollama generated {len(clean_queries)} queries: {clean_queries}")
                        return clean_queries
            except json.JSONDecodeError:
                print("DEBUG: Failed to parse JSON, falling back to text parsing")
                return parse_queries_from_text(generated_text, prompt, max_queries)
                
    except Exception as e:
        print(f"ERROR: Ollama failed: {e}")
    
    return [prompt]  # Fallback


def generate_queries_openai(prompt: str, context: str, max_queries: int, model: str = "gpt-3.5-turbo") -> list[str]:
    """Generate queries using OpenAI"""
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPEN_AI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system", 
                        "content": f"Generate {max_queries} focused search queries as JSON array. Return only the JSON array."
                    },
                    {
                        "role": "user", 
                        "content": f"User Question: {prompt}\nCompany Context: {context}"
                    }
                ],
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            queries = json.loads(content)
            
            if isinstance(queries, list):
                clean_queries = [q.strip() for q in queries if q.strip()][:max_queries]
                print(f"DEBUG: OpenAI generated {len(clean_queries)} queries: {clean_queries}")
                return clean_queries
                
    except Exception as e:
        print(f"ERROR: OpenAI failed: {e}")
    
    return [prompt]  # Fallback



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

def test_dynamic_mapping():
    """Test the dynamic model mapping with better error handling"""
    print("=== Testing Dynamic Model Mapping ===")
    
    # First check if Perplexica is accessible
    perplexica_running = check_perplexica_status()
    
    if not perplexica_running:
        print("\n🔄 Continuing with fallback mapping...")
    
    print("\n--- Testing Model Provider Detection ---")
    
    # Test some common models
    test_models = [
        ("llama3:latest", "chat"),
        ("gpt-3.5-turbo", "chat"),
        ("nomic-embed-text:latest", "embedding"),
        ("text-embedding-3-small", "embedding"),
        ("unknown-model", "chat")  # This should default to ollama
    ]
    
    for model, model_type in test_models:
        try:
            provider = get_model_provider(model, model_type)
            exists = validate_model_exists(model, model_type)
            status = "✅" if exists else "❌"
            print(f"{status} {model} ({model_type}) -> {provider} -> Exists: {exists}")
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")
    
    print("\n--- Available Models ---")
    print_available_models()
    
    if not perplexica_running:
        print("\n💡 To get live model data, start Perplexica and run: refresh_model_cache()")


def get_models_for_api() -> Dict:
    """
    Get models in format suitable for API responses
    """
    models = get_available_models_structured()
    
    response = {
        "chat_models": [],
        "embedding_models": [],
        "perplexica_status": _model_cache.get("perplexica_available", False)
    }
    
    for provider, provider_models in models["chat_models"].items():
        for model in provider_models:
            response["chat_models"].append({
                "name": model["name"],
                "display_name": model["display_name"],
                "provider": provider
            })
    
    for provider, provider_models in models["embedding_models"].items():
        for model in provider_models:
            response["embedding_models"].append({
                "name": model["name"],
                "display_name": model["display_name"], 
                "provider": provider
            })
    
    return response


if __name__ == "__main__":
    test_dynamic_mapping()
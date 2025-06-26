import requests
import json
import re

def extract_queries_from_av_brain(av_brain_prompt: str, av_brain_context: str, max_number_of_queries = 3) -> list[str]:
    """
    Use Llama 3 to generate focused search queries with extensive debugging
    """
    print(f"DEBUG: Starting query generation...")
    print(f"DEBUG: Input prompt: '{av_brain_prompt}'")
    print(f"DEBUG: Input context length: {len(av_brain_context)} characters")
    print(f"DEBUG: Max queries requested: {max_number_of_queries}")
    
    # Create a prompt for Llama 3 to generate search queries
    llm_prompt = f"""You are an expert at generating web search queries. Given a user's question and company context, generate {max_number_of_queries} focused search queries.

User Question: {av_brain_prompt}
Company Context: {av_brain_context}

Return your response as a valid JSON array of strings. Example format:
["query 1", "query 2", "query 3"]

Do not include any other text, just the JSON array:"""

    print(f"DEBUG: Generated LLM prompt (first 200 chars): {llm_prompt[:200]}...")

    try:
        print("DEBUG: Making request to Ollama API...")
        
        # Call Llama 3 via Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": llm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=30  # Add timeout
        )
        
        print(f"DEBUG: Ollama response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"DEBUG: Ollama response keys: {result.keys()}")
            
            generated_text = result.get("response", "").strip()
            print(f"DEBUG: Generated text: '{generated_text}'")
            print(f"DEBUG: Generated text length: {len(generated_text)}")
            
            # Try to parse as JSON
            try:
                print("DEBUG: Attempting JSON parsing...")
                queries = json.loads(generated_text)
                print(f"DEBUG: Parsed JSON successfully: {queries}")
                print(f"DEBUG: Type of parsed result: {type(queries)}")
                
                if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                    # Clean and filter queries
                    clean_queries = [q.strip() for q in queries if q.strip() and len(q.strip()) > 3]
                    clean_queries = clean_queries[:max_number_of_queries]
                    
                    if clean_queries:
                        print(f"DEBUG: Successfully generated {len(clean_queries)} clean queries: {clean_queries}")
                        return clean_queries
                    else:
                        print("DEBUG: No valid queries after cleaning")
                else:
                    print(f"DEBUG: JSON result is not a list of strings: {type(queries)}")
                    
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON parsing failed: {e}")
                print("DEBUG: Falling back to text parsing...")
                return parse_queries_from_text(generated_text, max_number_of_queries, av_brain_prompt)
                
        else:
            print(f"ERROR: Ollama API call failed with status {response.status_code}")
            print(f"ERROR: Response content: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Network error calling Ollama: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error in LLM query generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("DEBUG: Using fallback - returning original prompt")
    return [av_brain_prompt]


def parse_queries_from_text(text: str, max_queries: int, fallback_query: str) -> list[str]:
    """
    Robust text parsing for when JSON fails
    """
    print(f"DEBUG: Text parsing input: '{text}'")
    queries = []
    
    # Method 1: Try to extract JSON from the text
    json_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if json_match:
        print(f"DEBUG: Found JSON-like pattern: {json_match.group(0)}")
        try:
            json_content = '[' + json_match.group(1) + ']'
            parsed = json.loads(json_content)
            if isinstance(parsed, list):
                queries = [str(q).strip().strip('"\'') for q in parsed]
                print(f"DEBUG: Extracted queries from JSON pattern: {queries}")
        except Exception as e:
            print(f"DEBUG: Failed to parse JSON pattern: {e}")
    
    # Method 2: Line-by-line parsing
    if not queries:
        print("DEBUG: Trying line-by-line parsing...")
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            print(f"DEBUG: Processing line {i}: '{line}'")
            
            # Skip empty lines and common intro/outro phrases
            skip_phrases = [
                'here are', 'search queries', 'queries:', 'these queries',
                'should help', 'you can use', 'try searching', ''
            ]
            
            if any(phrase in line.lower() for phrase in skip_phrases):
                print(f"DEBUG: Skipping line {i} (contains skip phrase)")
                continue
                
            # Remove common prefixes
            original_line = line
            prefixes = [r'^\d+\.?\s*', r'^-\s*', r'^\*\s*', r'^•\s*', r'^["\']', r'["\']$']
            for prefix in prefixes:
                line = re.sub(prefix, '', line).strip()
            
            # Remove quotes at start/end
            line = line.strip('\'"')
            
            if len(line) > 3 and len(line) < 100:  # Reasonable query length
                queries.append(line)
                print(f"DEBUG: Added query from line {i}: '{original_line}' -> '{line}'")
    
    # Filter and limit
    queries = [q for q in queries if q and len(q) > 3][:max_queries]
    
    if not queries:
        print("WARNING: Could not parse any queries from LLM output, using fallback")
        return [fallback_query]
    
    print(f"DEBUG: Final parsed queries: {queries}")
    return queries


def test_ollama_connection():
    """
    Test if Ollama is running and accessible
    """
    print("DEBUG: Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"DEBUG: Ollama is running. Available models: {[m['name'] for m in models['models']]}")
            return True
        else:
            print(f"ERROR: Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        print("DEBUG: Make sure Ollama is running with: ollama serve")
        return False


def test_query_generation():
    """
    Test the query generation with a simple example
    """
    print("\n" + "="*50)
    print("TESTING QUERY GENERATION")
    print("="*50)
    
    # First test Ollama connection
    if not test_ollama_connection():
        print("ERROR: Ollama connection failed. Skipping LLM test.")
        return
    
    test_prompt = "What are the latest LLMs released?"
    test_context = "I'm a freelance data scientist and AI engineer, based in Italy and remote working for an american company."
    
    print(f"\nTesting with:")
    print(f"Prompt: {test_prompt}")
    print(f"Context: {test_context}")
    print("\n" + "-"*30)
    
    queries = extract_queries_from_av_brain(test_prompt, test_context, 3)
    
    print(f"\nFINAL RESULT: {queries}")
    print("="*50)


if __name__ == "__main__":
    test_prompt = "What are the latest LLMs released?"
    test_context = "I'm a data scientist"
    queries = extract_queries_from_av_brain(test_prompt, test_context)
    print(f"Generated queries: {queries}")
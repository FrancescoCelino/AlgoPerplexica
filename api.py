from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import main
from typing import Optional

app = FastAPI(title = "Perplexica Integration API",
              description = "API for integrating with Perplexica for AV Brain processing")

class AVBrainRequest(BaseModel):
    av_brain_prompt: str
    av_brain_context: str
    query_generator_model: Optional[str] = "llama3:latest"
    context_enrichment_model: Optional[str] = "llama3:latest"
    perplexica_embedding_model: Optional[str] = "nomic-embed-text:latest"
    max_queries: Optional[int] = 2

class AVBrainResponse(BaseModel): 
    enriched_context: str
    query_generation_used: str
    context_enrichment_used: str
    generated_queries: list[str]  

@app.post("/enrich", response_model=AVBrainResponse)
async def enrich_av_brain(request: AVBrainRequest):
    queries = main.extract_queries_from_av_brain(
        request.av_brain_prompt, 
        request.av_brain_context, 
        max_number_of_queries=request.max_queries, 
        model = request.query_generator_model
    )
    print(f"API generated queries: {queries}")
    enriched = main.enrich_context_from_queries(
        queries, 
        request.av_brain_context, 
        enriching_model=request.context_enrichment_model, 
        embedding_model=request.perplexica_embedding_model
    )
    return AVBrainResponse(
        enriched_context=enriched,
        query_generation_used=request.query_generator_model,
        context_enrichment_used=request.context_enrichment_model,
        generated_queries=queries
    )

    
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Perplexica Integration API, visit /docs for API documentation.",
            "endpoints": {
                "health": "/health",
                "enrich": "/enrich",
                "models": "/models",
                "docs": "/docs"
            }}

@app.get("/models")
async def get_models():
    try: 
        main.print_available_models()
        return {"status": "success", "message": "Models fetched successfully, see terminal for details."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
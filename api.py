from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import main

app = FastAPI(title = "Perplexica Integration API",
              description = "API for integrating with Perplexica for AV Brain processing")

class AVBrainRequest(BaseModel):
    av_brain_prompt: str
    av_brain_context: str

class AVBrainResponse(BaseModel): 
    enriched_context: str

@app.post("/enrich", response_model=AVBrainResponse)
async def enrich_av_brain(request: AVBrainRequest):
    queries = main.extract_queries_from_av_brain(request.av_brain_prompt, request.av_brain_context, 2)
    print = f"API generated queries: {queries}"
    enriched = main.enrich_context_from_queries(queries, request.av_brain_context)
    return AVBrainResponse(enriched_context=enriched)

    
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
        main.get_available_models()
        return {"status": "success", "message": "Models fetched successfully, see terminal for details."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
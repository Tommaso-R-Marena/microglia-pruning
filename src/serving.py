"""FastAPI app for model serving."""

from __future__ import annotations

from functools import lru_cache

from .inference import GenerationConfig, InferenceEngine


@lru_cache(maxsize=1)
def get_engine(model_name: str, backend: str) -> InferenceEngine:
    return InferenceEngine(model_name=model_name, backend=backend)


def create_app(model_name: str = "gpt2", backend: str = "hf"):
    """Create a FastAPI app.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI dependencies are not installed. Install with `pip install fastapi uvicorn`."
        ) from exc

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1)
        max_new_tokens: int = 128
        temperature: float = 0.0
        top_p: float = 1.0

    class GenerateResponse(BaseModel):
        text: str

    app = FastAPI(title="microglia-pruning-api", version="1.0.0")

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        engine = get_engine(model_name=model_name, backend=backend)
        cfg = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        text = engine.generate(request.prompt, config=cfg)
        return GenerateResponse(text=text)

    return app

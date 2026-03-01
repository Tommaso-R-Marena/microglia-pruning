import sys
import types

import pytest

from src.serving import create_app


class FakeApp:
    def __init__(self, **kwargs):
        self.routes = {}

    def get(self, path):
        def decorator(func):
            self.routes[("GET", path)] = func
            return func
        return decorator

    def post(self, path, response_model=None):
        def decorator(func):
            self.routes[("POST", path)] = func
            return func
        return decorator


class DummyEngine:
    def __init__(self, model_name: str, backend: str):
        self.model_name = model_name
        self.backend = backend

    def generate(self, prompt, config=None):
        return f"generated:{prompt}"


def install_fake_fastapi(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    setattr(fastapi_mod, "FastAPI", FakeApp)

    pydantic_mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default=None, **kwargs):
        return default

    setattr(pydantic_mod, "BaseModel", BaseModel)
    setattr(pydantic_mod, "Field", Field)

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)


def test_create_app_without_fastapi(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.setitem(sys.modules, "pydantic", None)
    with pytest.raises(RuntimeError):
        create_app()


def test_create_app_routes(monkeypatch):
    install_fake_fastapi(monkeypatch)
    monkeypatch.setattr("src.serving.InferenceEngine", DummyEngine)
    app = create_app(model_name="dummy", backend="hf")
    assert ("GET", "/health") in app.routes
    assert ("POST", "/generate") in app.routes

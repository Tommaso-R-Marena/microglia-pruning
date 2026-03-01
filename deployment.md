# Deployment Guide

## Architecture

- **Training**: `scripts/train.py` with `--precision {fp32,fp16,bf16}`
- **Inference**: `src.inference.InferenceEngine` with `hf` or `vllm` backend
- **Serving**: FastAPI via `scripts/serve_api.py`
- **Interop**: ONNX export via `scripts/export_onnx.py`

## Docker (example)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -e .
EXPOSE 8000
CMD ["python", "scripts/serve_api.py", "--model", "gpt2", "--backend", "hf", "--host", "0.0.0.0", "--port", "8000"]
```

## Kubernetes Notes

- Run API with readiness probe on `/health`
- Use HPA on request latency / CPU / GPU utilization
- For vLLM, allocate one pod per GPU and tune tensor parallelism

## Benchmarking Procedure

1. Warm up with 10 requests.
2. Run 200 requests for each backend.
3. Track requests/sec and p95 latency.
4. Record GPU memory and utilization.

Command template:

```bash
python scripts/perf_benchmark.py --model gpt2 --backend hf --requests 200
python scripts/perf_benchmark.py --model gpt2 --backend vllm --requests 200
```

## Observability

- Add request/latency logging middleware in FastAPI.
- Export metrics to Prometheus.
- Alert on p95 latency regression.

# Performance Benchmarks

This file documents the benchmark workflow for production validation.

## Command

```bash
python scripts/perf_benchmark.py --model gpt2 --backend hf --requests 20
python scripts/perf_benchmark.py --model gpt2 --backend vllm --requests 20
```

## Example Results Template

| Backend | Requests | Elapsed (s) | Requests/s |
|---|---:|---:|---:|
| HF | 20 | _fill after run_ | _fill after run_ |
| vLLM | 20 | _fill after run_ | _fill after run_ |

## Production Benchmark Recommendations

- Run with realistic prompt length distribution.
- Measure p50/p95 latency and throughput at concurrency levels 1, 8, 32, 64.
- Include cold-start and warm-start behavior.
- Compare GPU memory footprint for HF vs vLLM.

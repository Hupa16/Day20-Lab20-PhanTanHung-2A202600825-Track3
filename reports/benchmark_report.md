# Benchmark Report

## Link Trace

- LangSmith project: `Nhat QUang`
- Multi-agent workflow trace: https://smith.langchain.com/o/032731e8-1f5e-40a7-96c4-006fd89cd206/projects/p/70f2be3e-c782-49d7-af5e-a4227ec48561/r/019efa26-f00d-7d40-84de-3e94c81ddbf7?trace_id=019efa26-f00d-7d40-84de-3e94c81ddbf7&start_time=2026-06-24T15:01:56.365037

| Run | Latency (s) | Cost (USD) | Input tok | Output tok | Quality | Citations | Failures | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 19.07 |  | 58 | 371 | 5.0 |  | 0% | routes=; sources=0; answer_words=282 |
| multi-agent | 19.82 |  | 2420 | 987 | 9.6 | 80% | 0% | routes=researcher>analyst>writer>done; sources=5; answer_words=245 |

## Notes

- Quality is a lightweight heuristic for lab comparison, not a substitute for peer review.
- Citation coverage counts unique source markers like `[1]` in the final answer.
- Failure rate is `100%` when a run records errors or produces no final answer.

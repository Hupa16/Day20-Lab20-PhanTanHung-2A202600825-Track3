# Failure Modes And Fixes

## Search provider rejects the request

Symptom: `Tavily search failed with HTTP 401` or a network timeout.

Fix: verify `TAVILY_API_KEY`, keep `SearchClient` errors scoped to the researcher step, and let the workflow record `workflow.failed` with the failing route.

## LLM provider is unavailable

Symptom: `LLM completion failed` from OpenAI or missing `OPENAI_API_KEY`.

Fix: keep provider calls inside `LLMClient`, use SDK retries/timeouts there, and surface failures through `state.errors` instead of letting agents run forever.

## Supervisor loops forever

Symptom: route history grows without producing `final_answer`.

Fix: `SupervisorAgent` stops when `MAX_ITERATIONS` is reached and records `max_iterations_reached` in trace payloads.

## Weak citations

Symptom: final answer has unsupported claims or missing source markers.

Fix: Writer prompt requires citation markers, and benchmark citation coverage counts unique `[n]` references against retrieved sources.

## Trace upload fails

Symptom: LangSmith network/auth failure, but local workflow output still exists.

Fix: `trace_span` falls back to local timing and `ResearchState.trace`; remote observability should not break the main workflow.

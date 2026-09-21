# Real-time data

Real-time context is inference-only, never training data. Enable it only with `ENABLE_REALTIME_DATA=true` and an approved `REALTIME_DATA_URL`. The endpoint receives `q=<prompt>` and must return JSON with a string list: `{ "facts": ["…"] }`.

If the provider is missing, unreachable, malformed, or the prompt is not a real-time request, no facts are injected and the job metadata records `realtime_data: null`. The system therefore never presents sample, stale, or unverified market values as current information.

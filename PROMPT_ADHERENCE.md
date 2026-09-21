# Prompt adherence

Every request now receives a deterministic constraint plan before keyframe and video generation. It preserves the requested subject (including modifiers such as `red sports car`), action, environment, weather, time, camera, style, and a dynamic negative prompt in every scene's `metadata`.

For example, the rainy red-car request carries `red sports car`, `driving`, `highway`, `heavy rain`, `night`, and `smooth tracking shot`, while explicitly excluding a dry road, daylight, a parked vehicle, and a wrong vehicle color.

This is prompt control, not semantic validation. `PROMPT_ADHERENCE_PROVIDER=disabled` is intentionally the default because no VLM evaluator is configured. Do not interpret existing heuristic training metrics as a prompt-adherence benchmark. Configure a real VLM evaluator, generate a held-out before/after set, and retain its reports before claiming an accuracy improvement or enabling automatic retries.

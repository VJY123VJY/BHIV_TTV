# Model providers

The system composes specialized adapters: LLM (story/scenes), image (keyframes), video (motion), TTS (dialogue), lip sync (facial motion), vision consistency (identity/prompt lock), and FFmpeg (assembly). Select video through `VIDEO_PROVIDER=opencv|external|wan`; select TTS through `TTS_PROVIDER=auto|edge|google|xtts|elevenlabs|azure|local`.

`opencv` and `viseme` are development fallbacks. `wan` is a real GPU inference adapter and is loaded lazily. External providers need their own credentials/configuration and should surface errors rather than silently claim generation succeeded.

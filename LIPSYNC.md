# Lip synchronization

`BaseLipSyncAdapter` keeps lip sync replaceable. The router supports `musetalk`, `wav2lip`, and the CPU-friendly `viseme` fallback. Each adapter implements `sync_clip(video_path, audio_path, output_path)`; Wav2Lip and MuseTalk only run their neural path when an installed checkpoint is configured and otherwise report their fallback in scene metadata.

Set `LIPSYNC_PROVIDER=musetalk` and `MUSETALK_CHECKPOINT_PATH` (or use `wav2lip` with `LIPSYNC_MODEL_PATH`) on a GPU worker. The fallback creates approximate viseme animation for local development; it is not a claim of phoneme-accurate neural lip synchronization.

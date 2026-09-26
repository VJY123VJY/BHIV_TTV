# TTV Studio: legal data and measurable-improvement pipeline

## Current audit

| Area | Finding |
| --- | --- |
| Production backbone | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, loaded as Diffusers `WanImageToVideoPipeline` when `VIDEO_PROVIDER=wan`. It is a 5B hybrid text+image-to-video video-diffusion model with a high-compression video VAE; the Wan 2.2 family adds MoE innovations. |
| Local default | `VIDEO_PROVIDER=opencv`; a procedural development fallback, not a learned TTV model. |
| Legacy research model | `SpatialTemporalTTVModel`: small 3D-convolution/cross-attention 128x128/8-frame model with custom LoRA. Its checkpoints are **not compatible with Wan**. |
| Conditioning | Wan uses a scene keyframe plus enriched prompt. Reference video currently becomes one still, so native video-to-video conditioning is not implemented. |
| Fine-tuning | Legacy custom LoRA is supported. `MODEL_MODE=wan_lora` now loads a Wan-compatible Diffusers LoRA from `WAN_LORA_PATH`; it rejects missing/incompatible adapters rather than silently loading a legacy checkpoint. |
| Prompt flow | validate → LLM analysis → constraints/negative prompt → story/scenes → keyframe → video. |
| Hardware | Windows 10; 4 physical/8 logical CPU cores; 15.65 GB RAM (2.74 GB free at audit); PyTorch `2.13.0+cpu`; no CUDA GPU/VRAM. Wan inference and practical video LoRA training cannot run locally. |

Use an NVIDIA CUDA system for Wan. The supplied 720p template declares 24 GB minimum VRAM, but validate the actual recipe/model memory profile before launch. A completed legacy training run does not demonstrate production-model improvement.

## Dataset policy and source registry

The canonical machine-readable registry is `data/dataset_registry.json`. It contains the source URL, license treatment, commercial/derivative status, count, resolution/duration scope, categories and attribution requirements required for each source. Dynamic counts are measured in the acquisition manifest, never invented.

| Source | Collection status | Terms |
| --- | --- | --- |
| Wikimedia Commons | API collection allowed only with verified per-item creator, page URL, license URL and CC0/public domain/CC BY/CC BY-SA license | Commercial derivatives allowed under those licenses; retain attribution and share-alike requirements. |
| NASA Image and Video Library | Manual review only | Generally public-domain US government material, but third-party media and NASA identifiers have exceptions. Never bulk download. |
| Internet Archive | Manual review only | Hosting is not a license; verify original rights holder and item-level license. |
| User-authorized media | Local import with attestation | Requires a `rights_attestation_id`; keep the release outside public metadata. |

YouTube, Instagram, Facebook, X/Twitter and TikTok are blocked in the registry. There is no generic remote-URL downloader.

## Dataset contract

```text
data/
  raw/ validated/ clips/ frames/ captions/ processed/
  train/metadata.jsonl validation/metadata.jsonl test/metadata.jsonl
  metadata/ rejected/ dataset_registry.json dataset_report.json
models/base/ models/experiments/ models/checkpoints/ models/production/
```

Each row needs source URL, creator, license, license URL, source group/parent, SHA-256, quality metrics, orientation, caption/status, category, style and motion. VLM captions are always `needs_review`; only human-approved captions enter `processed/metadata.jsonl`. Captions never come from filenames, queries or URLs.

```text
reviewed local manifest → provenance/decode/quality gate → shot-aware clips
→ sampled frames → VLM temporal caption → human review → deduplicate
→ source-grouped 70/15/15 split → train/evaluate
```

Clips are normalized at 24 FPS using contain-with-padding, never stretching. Run separate 1280x720 landscape and 720x1280 portrait collections; `aspect_ratio` is retained. Metrics include resolution, FPS, duration, blur, brightness, motion, sampled frame hashes, decode evidence and quality score. Watermark/logo review remains manual until an explicit OCR/logo detector is configured.

## Commands

Run at the repository root. The collector only enables registry-gated Wikimedia acquisition:

```powershell
python -m scripts.collect_dataset --query "dog running beach" --media-type video --license CC-BY --limit 20 --min-resolution 1280x720
python -m scripts.validate_dataset --input data/reviewed_assets.jsonl --output data/validated/assets.jsonl --min-width 1280 --min-height 720
python -m scripts.extract_clips --input data/validated/assets.jsonl --output data/clips/metadata.jsonl --fps 24 --width 1280 --height 720 --seconds 4
python -m scripts.extract_frames --input data/clips/metadata.jsonl --output data/frames/metadata.jsonl --count 8
python -m scripts.generate_captions --input data/frames/metadata.jsonl --output data/captions/metadata.jsonl --model HuggingFaceTB/SmolVLM-256M-Instruct
# Human review: set caption_status=approved only for factual captions.
python -m scripts.build_metadata --input data/captions/reviewed_metadata.jsonl --output data/processed/metadata.jsonl
python -m scripts.detect_duplicates --input data/processed/metadata.jsonl --output data/processed/deduplicated_metadata.jsonl
python -m scripts.create_splits --input data/processed/deduplicated_metadata.jsonl --data-root data --dataset-version dataset_v001
python -m scripts.evaluate_dataset --input data/processed/deduplicated_metadata.jsonl --output data/dataset_report.json --dataset-version dataset_v001
```

The reviewed local manifest must include `id`, `source_id`, `local_path`, `source_url`, `creator`, `license`, `license_url`, `category`, `style`, `motion`, and `source_group_id`; `user_authorized` additionally needs `rights_attestation_id`.

The legacy smoke trainer is for the repository research model only, **not Wan**:

```powershell
python -m scripts.train_lora --config training/configs/lora_finetune.yaml --version-name ttv_legacy_lora_v002
```

For a production Wan LoRA, provision CUDA, use `training/configs/wan_lora_template.yaml` with a maintained Wan-compatible Diffusers trainer, retain artifacts under `models/experiments/ttv_wan_lora_v001`, and promote only after held-out evaluation. Configure inference:

```powershell
$env:VIDEO_PROVIDER="wan"
$env:MODEL_MODE="wan_lora"
$env:WAN_LORA_PATH="models/experiments/ttv_wan_lora_v001"
python -m uvicorn backend.app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Use `MODEL_MODE=base` for base-model comparison; local development uses `VIDEO_PROVIDER=opencv`.

The six fixed prompts are in `data/benchmarks/ttv_fixed_v001.json`:

```powershell
python -m scripts.run_benchmark --base-dir generated/benchmark/base --fine-tuned-dir generated/benchmark/fine_tuned --output training/evaluation/reports/fixed_benchmark_comparison.json
```

Keep prompt ID, seed (where supported), duration, FPS, resolution, scheduler and inference steps equal. The report separates automatic frame metrics from a shared 1–5 rubric for prompt adherence, motion, temporal consistency, visual quality and style; aspect ratio is pass/fail. It deliberately says `not established` until real base/fine-tuned outputs and ratings exist.

## Limitations

- No benchmark evidence currently shows the legacy checkpoint improves Wan; it is not promoted.
- This workstation needs external CUDA capacity, compatible PyTorch/drivers, storage and a maintained Wan LoRA recipe.
- Per-asset licenses may impose attribution/share-alike duties and do not eliminate personality, trademark, privacy or consent risks; obtain legal review for commercial deployment.
- NASA/Internet Archive require individual review; generic social/copyrighted sources remain unsupported.
- VLM captions and automatic style/motion labels require review. Watermark/logo detection needs an explicit external model.
- LLM expansion, TTS, image generation and optional lip-sync providers may require external API keys or model weights.

# Dataset pipeline

Dataset preparation is deliberately independent of video generation. Only assets with verifiable provenance and an approved license can enter a training manifest.

## Sources and policy

The built-in source adapters query Wikimedia Commons and Internet Archive. They retain the source item URL, direct download URL, creator, license label, license URL, retrieval time, SHA-256, media type, resolution, duration, and dataset version. Synthetic and user-owned assets must be labelled `internal-synthetic` or `user-authorized`; user uploads require documented ownership/consent outside this repository.

Unknown, non-commercial, no-derivatives, or all-rights-reserved licenses are rejected. The ingest command never crawls arbitrary sites, and a source-level search plus an explicit item limit bounds acquisition.

```powershell
python -m dataset.ingest --source wikimedia --query "people speaking classroom" --limit 50 --license CC-BY --min-resolution 1280x720 --media-type image
python -m dataset.ingest --source archive_org --query "public domain farming" --limit 20 --license "public domain" --media-type video
python -m dataset.clean
python -m dataset.deduplicate
python -m dataset.caption
python -m dataset.manifest
```

Downloads resume via `.part` files and are subject to the configured size cap. Data moves through `data/raw`, `staging`, `processed`, `rejected`, `manifests`, `embeddings`, `captions`, and `metadata`; category subsets live under `datasets/`.

Image checks include decode integrity, resolution, blur, black frame, SHA-256, and perceptual hash. Video inspection records FPS, duration, dimensions, scenes/keyframes, and configurable 2/4/6/8/12-second clips. `dataset_report.json` and split JSONL manifests are generated after review. Split before training; never add evaluation/test records to a training command.

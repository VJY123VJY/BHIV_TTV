# Training

Training is opt-in (`ENABLE_DATASET_TRAINING=true`) and is not triggered by a generation request or real-time fact lookup. Start with a licensed, reviewed manifest and fine-tune adapters/LoRAs rather than training a video foundation model from scratch.

```powershell
python -m training.prepare --manifest data/manifests/train.jsonl --output data/staging
python -m training.train_lora --config training/configs/lora_finetune.yaml --version-name farm_v1
python -m training.evaluate --config training/configs/lora_finetune.yaml --checkpoint <checkpoint-path>
```

A checkpoint file alone is not evidence of model quality. Record the manifest version, config, hardware, validation metrics, held-out test results, and generated artifacts before labelling a model fine-tuned or production-ready.

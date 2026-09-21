"""
CLI for dataset perceptual & cryptographic deduplication.
Usage:
  python -m dataset.deduplicate
"""
import json
from pathlib import Path
from dataset.quality import hamming_distance


def main():
    print("=== Scanning Dataset for Duplicates (SHA-256 + pHash) ===")
    meta_dir = Path("data/metadata")
    rej_dir = Path("data/rejected")
    rej_dir.mkdir(parents=True, exist_ok=True)

    seen_sha = {}
    seen_phash = {}
    dupes = 0

    for meta_file in sorted(meta_dir.glob("*.json")):
        try:
            data = json.loads(meta_file.read_text())
            sha = data.get("sha256")
            phash = data.get("phash")
            asset_id = data.get("asset_id")

            is_dupe = False
            if sha in seen_sha:
                is_dupe = True
                reason = f"Exact SHA256 match with {seen_sha[sha]}"
            elif phash:
                for prior_phash, prior_id in seen_phash.items():
                    if hamming_distance(phash, prior_phash) <= 3:
                        is_dupe = True
                        reason = f"Perceptual pHash similarity with {prior_id}"
                        break

            if is_dupe:
                dupes += 1
                rej_path = rej_dir / f"{asset_id}_duplicate.json"
                rej_path.write_text(json.dumps({"asset_id": asset_id, "reason": reason}))
                meta_file.unlink()
                media_path = Path(data.get("local_path", ""))
                if media_path.exists():
                    media_path.unlink()
            else:
                if sha:
                    seen_sha[sha] = asset_id
                if phash:
                    seen_phash[phash] = asset_id

        except Exception as e:
            continue

    print(f"Deduplication complete: {dupes} duplicates detected and quarantined.")


if __name__ == "__main__":
    main()

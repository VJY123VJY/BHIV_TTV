"""
Internet Archive open media dataset source adapter.
"""
from typing import List, Dict, Any
import httpx
from dataset.quality import verify_license


class ArchiveOrgSource:
    """Ingests public-domain and CC-licensed media from Internet Archive (archive.org)."""

    SEARCH_URL = "https://archive.org/advancedsearch.php"

    def search_assets(
        self,
        query: str,
        limit: int = 20,
        media_type: str = "movies",  # movies | image
    ) -> List[Dict[str, Any]]:
        headers = {"User-Agent": "BHIV-TTV-DatasetEngine/2.0 (research-education)"}
        med_type = "movies" if media_type == "video" else "image"
        # Search query filtering for creative commons / public domain
        q = f'{query} AND mediatype:{med_type} AND (licenseurl:*creativecommons* OR licenseurl:*publicdomain*)'
        params = {
            "q": q,
            "fl[]": "identifier,title,creator,licenseurl,mediatype",
            "rows": min(limit, 50),
            "output": "json",
        }

        try:
            with httpx.Client(timeout=15.0, headers=headers) as client:
                res = client.get(self.SEARCH_URL, params=params)
                if res.status_code != 200:
                    return []
                docs = res.json().get("response", {}).get("docs", [])
                results = []
                for doc in docs:
                    lic_url = doc.get("licenseurl", "")
                    lic_info = verify_license(lic_url)
                    if not lic_info.training_eligible:
                        continue

                    ident = doc.get("identifier")
                    # An item details page is not a training asset. Resolve one
                    # actual media file and retain the item-level license URL as
                    # provenance.  If this lookup fails, safely skip it.
                    try:
                        metadata = client.get(f"https://archive.org/metadata/{ident}").json()
                        allowed = (".mp4", ".webm", ".mov") if media_type == "video" else (".jpg", ".jpeg", ".png", ".webp")
                        file_info = next(
                            (f for f in metadata.get("files", []) if str(f.get("name", "")).lower().endswith(allowed)),
                            None,
                        )
                        if not file_info:
                            continue
                        download_url = f"https://archive.org/download/{ident}/{file_info['name']}"
                    except (httpx.HTTPError, ValueError, KeyError):
                        continue
                    results.append({
                        "asset_id": f"ia_{ident}",
                        "title": doc.get("title"),
                        "source_name": "archive_org",
                        "source_url": f"https://archive.org/details/{ident}",
                        "download_url": download_url,
                        "license": lic_info.license,
                        "license_url": lic_url,
                        "creator": doc.get("creator"),
                        "allowed_for_training": lic_info.training_eligible,
                        "media_type": media_type,
                        "category": "general",
                    })
                return results
        except Exception:
            return []


archive_org_source = ArchiveOrgSource()

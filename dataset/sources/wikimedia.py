"""
Wikimedia Commons public-domain & Creative Commons dataset source adapter.
"""
from typing import List, Dict, Any, Optional
import httpx
from dataset.quality import verify_license
from dataset.registry import license_is_allowed


class WikimediaSource:
    """Ingests public-domain and Creative Commons assets from Wikimedia Commons."""

    API_URL = "https://commons.wikimedia.org/w/api.php"

    def search_assets(
        self,
        query: str,
        limit: int = 20,
        media_type: str = "image",
    ) -> List[Dict[str, Any]]:
        """
        Queries Wikimedia Commons API for media assets with license metadata.
        Returns candidates with verified licensing.
        """
        headers = {"User-Agent": "BHIV-TTV-DatasetEngine/2.0 (research-education)"}
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} filetype:{'bitmap' if media_type == 'image' else 'video'}",
            "gsrlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|sha1",
            "format": "json",
        }

        try:
            with httpx.Client(timeout=15.0, headers=headers) as client:
                res = client.get(self.API_URL, params=params)
                if res.status_code != 200:
                    return []
                data = res.json()
                pages = data.get("query", {}).get("pages", {})
                results = []
                for _, page in pages.items():
                    info = page.get("imageinfo", [{}])[0]
                    ext_meta = info.get("extmetadata", {})
                    license_name = ext_meta.get("LicenseShortName", {}).get("value", "unknown")
                    license_info = verify_license(license_name)

                    # Only retain assets permitted for model training
                    # Require the explicit registry allowlist *and* sufficient
                    # attribution metadata; neither Commons hosting nor a
                    # generic free-license label is enough by itself.
                    creator = ext_meta.get("Artist", {}).get("value", "").strip()
                    license_url = ext_meta.get("LicenseUrl", {}).get("value", "").strip()
                    if not license_info.training_eligible or not license_is_allowed(license_name) or not creator or not license_url:
                        continue

                    results.append({
                        "asset_id": f"wiki_{page.get('pageid')}",
                        "title": page.get("title"),
                        "source_name": "wikimedia",
                        "source_url": info.get("url"),
                        "download_url": info.get("url"),
                        "license": license_info.license,
                        "license_url": license_url,
                        "creator": creator,
                        "width": info.get("width"),
                        "height": info.get("height"),
                        "allowed_for_training": license_info.training_eligible,
                        "media_type": media_type,
                        "category": "general",
                    })
                return results
        except Exception:
            return []


wikimedia_source = WikimediaSource()

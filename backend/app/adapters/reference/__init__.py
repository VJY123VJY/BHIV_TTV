"""Reference source adapter factory."""
from urllib.parse import urlparse

from app.adapters.reference.direct_media_adapter import DirectMediaAdapter
from app.adapters.reference.social_adapter import SocialMediaAdapter
from app.adapters.reference.validator import detect_source


def get_reference_adapter(url: str):
    parsed = urlparse(url)
    source = detect_source(parsed.hostname or "")
    if source in SocialMediaAdapter.HANDLED:
        return SocialMediaAdapter(source)
    return DirectMediaAdapter()

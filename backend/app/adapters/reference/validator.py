"""URL validation, source detection, and SSRF protections for reference media."""
from __future__ import annotations

import ipaddress
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

from app.core.exceptions import ValidationError

ALLOWED_SCHEMES = {"http", "https"}
MAX_URL_LENGTH = 2048

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google.com",
    "instance-data",
}

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}
FACEBOOK_HOSTS = {"facebook.com", "www.facebook.com", "m.facebook.com", "fb.watch", "www.fb.watch"}
X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}


class ReferenceValidationError(ValidationError):
    pass


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_reserved
    )


def detect_source(hostname: str) -> str:
    host = (hostname or "").lower().rstrip(".")
    if host in YOUTUBE_HOSTS:
        return "youtube"
    if host in INSTAGRAM_HOSTS:
        return "instagram"
    if host in FACEBOOK_HOSTS:
        return "facebook"
    if host in X_HOSTS:
        return "x"
    return "direct"


def validate_public_url(url: Optional[str], resolve_dns: bool = True) -> Tuple[str, str]:
    if not url or not str(url).strip():
        raise ReferenceValidationError("Reference URL is empty.")
    raw = str(url).strip()
    if len(raw) > MAX_URL_LENGTH:
        raise ReferenceValidationError("Reference URL is too long.")
    if any(ch.isspace() for ch in raw):
        raise ReferenceValidationError("Reference URL contains invalid whitespace.")

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ReferenceValidationError("Only HTTP and HTTPS reference URLs are allowed.")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ReferenceValidationError("Reference URL is missing a hostname.")
    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".local") or hostname.endswith(".internal"):
        raise ReferenceValidationError("Private or internal hostnames are not allowed.")
    if parsed.username or parsed.password:
        raise ReferenceValidationError("URLs with embedded credentials are not allowed.")
    if parsed.port in {22, 25, 3306, 5432, 6379, 11211, 27017}:
        raise ReferenceValidationError("This URL port is not allowed for media retrieval.")

    try:
        ipaddress.ip_address(hostname)
        if _is_private_ip(hostname):
            raise ReferenceValidationError("Private IP addresses are not allowed.")
    except ValueError:
        pass

    if resolve_dns:
        try:
            infos = socket.getaddrinfo(hostname, parsed.port or (443 if scheme == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ReferenceValidationError("The reference URL hostname could not be resolved.") from exc
        for info in infos:
            ip_str = info[4][0]
            if _is_private_ip(ip_str):
                raise ReferenceValidationError("The reference URL resolves to a private or internal address.")

    source = detect_source(hostname)
    return raw, source

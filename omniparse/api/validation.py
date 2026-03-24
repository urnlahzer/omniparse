"""Input validation functions for API endpoints -- SSRF prevention and file checks.

Pure validation functions with no side effects. Called by the /submit endpoint
before spawning the pipeline to fail fast on invalid input.

SSRF prevention (D-01, D-02, D-04):
  - validate_callback_url() enforces HTTPS, port 443, public IPs only

File validation (D-06, D-07):
  - validate_upload() enforces size limit and PDF magic bytes
"""
import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB (D-06)
PDF_MAGIC = b"%PDF-"


def validate_callback_url(url: str) -> str:
    """Validate a webhook callback URL against SSRF attack vectors.

    Enforces:
    - HTTPS scheme only (D-01)
    - Port 443 only (D-01, Pitfall 3 -- None means default/implied 443)
    - No userinfo credentials in URL
    - Hostname must resolve to public (non-private/reserved/loopback/link-local) IPs (D-02)

    Args:
        url: The callback URL to validate.

    Returns:
        The validated URL string, unchanged.

    Raises:
        ValueError: If the URL fails any validation check.
    """
    parsed = urlparse(url)

    # D-01: HTTPS only
    if parsed.scheme != "https":
        raise ValueError("Callback URL must use HTTPS")

    # D-01 + Pitfall 3: Port must be 443 (None = implied default = OK)
    if parsed.port is not None and parsed.port != 443:
        raise ValueError("Callback URL must use port 443")

    # Hostname presence
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Callback URL has no hostname")

    # Userinfo (SSRF bypass vector)
    if parsed.username or parsed.password:
        raise ValueError("Callback URL must not contain credentials")

    # D-02: Resolve hostname and check all IPs
    try:
        addr_infos = socket.getaddrinfo(hostname, 443)
    except OSError as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname} ({exc})")

    for family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
            logger.warning(
                "SSRF blocked: %s resolved to private/reserved IP %s", hostname, ip_str
            )
            raise ValueError(
                f"Callback URL resolves to private/reserved IP: {ip_str}"
            )

    return url


def validate_upload(file_bytes: bytes, filename: str) -> None:
    """Validate uploaded file size and format before pipeline processing.

    Enforces:
    - File size <= MAX_UPLOAD_BYTES (D-06)
    - PDF files must start with %PDF- magic bytes (D-07)

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename from the upload.

    Raises:
        ValueError: If the file exceeds the size limit or is a fake PDF.
    """
    # D-06: Size limit
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit"
        )

    # D-07: PDF magic byte check (case-insensitive extension match)
    if filename.lower().endswith(".pdf"):
        if not file_bytes[:5] == PDF_MAGIC:
            raise ValueError("File claims to be PDF but lacks PDF signature")

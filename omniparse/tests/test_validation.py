"""Tests for SSRF prevention and file upload validation functions.

Covers callback URL validation (SSRF-01) and file upload validation (INPT-01):
- HTTPS-only enforcement
- Port 443-only enforcement
- Private/reserved/loopback/link-local IP rejection
- Userinfo in URL rejection
- Unresolvable hostname rejection
- File size limit (100 MB)
- PDF magic byte verification
"""
from unittest.mock import patch

import pytest

from omniparse.api.validation import (
    MAX_UPLOAD_BYTES,
    PDF_MAGIC,
    validate_callback_url,
    validate_upload,
)


# ---------------------------------------------------------------------------
# Helper: mock DNS resolution to return specific IPs
# ---------------------------------------------------------------------------

def _mock_getaddrinfo(ip_str: str):
    """Return a mock getaddrinfo function that resolves to the given IP."""
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(2, 1, 6, "", (ip_str, port))]
    return fake_getaddrinfo


# ===========================================================================
# validate_callback_url — SSRF prevention (D-01, D-02, D-04)
# ===========================================================================


class TestCallbackUrlScheme:
    """D-01: Only HTTPS allowed."""

    def test_http_scheme_rejected(self):
        with pytest.raises(ValueError, match="Callback URL must use HTTPS"):
            validate_callback_url("http://example.com")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="Callback URL must use HTTPS"):
            validate_callback_url("ftp://example.com/hook")


class TestCallbackUrlPort:
    """D-01: Only port 443 allowed."""

    def test_non_443_port_rejected(self):
        with pytest.raises(ValueError, match="Callback URL must use port 443"):
            validate_callback_url("https://example.com:8443/hook")

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("93.184.216.34"))
    def test_explicit_443_accepted(self, _mock):
        result = validate_callback_url("https://example.com:443/hook")
        assert result == "https://example.com:443/hook"

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("93.184.216.34"))
    def test_implied_443_accepted(self, _mock):
        """Port=None (default) is implied 443 and must be accepted."""
        result = validate_callback_url("https://example.com/hook")
        assert result == "https://example.com/hook"


class TestCallbackUrlHostname:
    """Hostname validation."""

    def test_empty_hostname_rejected(self):
        with pytest.raises(ValueError, match="Callback URL has no hostname"):
            validate_callback_url("https:///hook")

    def test_userinfo_rejected(self):
        with pytest.raises(ValueError, match="Callback URL must not contain credentials"):
            validate_callback_url("https://user:pass@host.com/hook")

    def test_username_only_rejected(self):
        with pytest.raises(ValueError, match="Callback URL must not contain credentials"):
            validate_callback_url("https://user@host.com/hook")


class TestCallbackUrlDnsResolution:
    """D-02: Reject private/reserved/loopback/link-local IPs."""

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("127.0.0.1"))
    def test_loopback_rejected(self, _mock):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_callback_url("https://localhost/hook")

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("10.0.0.1"))
    def test_private_10_rejected(self, _mock):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_callback_url("https://internal.corp/hook")

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("192.168.1.1"))
    def test_private_192_168_rejected(self, _mock):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_callback_url("https://router.local/hook")

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("172.16.0.1"))
    def test_private_172_16_rejected(self, _mock):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_callback_url("https://docker.internal/hook")

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("169.254.169.254"))
    def test_link_local_rejected(self, _mock):
        """169.254.x.x is link-local (AWS metadata endpoint)."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_callback_url("https://metadata.internal/hook")

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("::1"))
    def test_ipv6_loopback_rejected(self, _mock):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_callback_url("https://ipv6host.example.com/hook")

    def test_unresolvable_hostname_rejected(self):
        with patch("omniparse.api.validation.socket.getaddrinfo",
                   side_effect=OSError("Name or service not known")):
            with pytest.raises(ValueError, match="Cannot resolve hostname"):
                validate_callback_url("https://does-not-exist.invalid/hook")


class TestCallbackUrlAccepted:
    """Valid callback URLs must pass through unchanged."""

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("93.184.216.34"))
    def test_public_https_url_accepted(self, _mock):
        url = "https://public-host.example.com/hook"
        assert validate_callback_url(url) == url

    @patch("omniparse.api.validation.socket.getaddrinfo",
           side_effect=_mock_getaddrinfo("93.184.216.34"))
    def test_public_https_with_path_accepted(self, _mock):
        url = "https://api.example.com/webhooks/omniparse"
        assert validate_callback_url(url) == url


# ===========================================================================
# validate_upload — file size and PDF magic byte checks (D-06, D-07)
# ===========================================================================


class TestUploadSizeLimit:
    """D-06: Reject files over 100 MB."""

    def test_oversized_file_rejected(self):
        big = b"x" * (MAX_UPLOAD_BYTES + 1)
        with pytest.raises(ValueError, match="limit"):
            validate_upload(big, "document.pdf")

    def test_exactly_at_limit_accepted(self):
        data = b"%" + b"P" + b"D" + b"F" + b"-" + b"\x00" * (MAX_UPLOAD_BYTES - 5)
        # Should not raise (exactly at limit, valid PDF header)
        validate_upload(data, "document.pdf")

    def test_under_limit_accepted(self):
        data = b"%PDF-1.4 some content"
        validate_upload(data, "document.pdf")


class TestUploadPdfMagic:
    """D-07: PDF files must start with %PDF- magic bytes."""

    def test_fake_pdf_rejected(self):
        with pytest.raises(ValueError, match="PDF signature"):
            validate_upload(b"NOT-A-PDF-FILE-CONTENT", "doc.pdf")

    def test_valid_pdf_accepted(self):
        validate_upload(b"%PDF-1.4 rest of content", "report.pdf")

    def test_non_pdf_skips_magic_check(self):
        """Non-PDF files should not be checked for PDF magic bytes."""
        validate_upload(b"PNG image bytes", "image.png")

    def test_uppercase_extension_triggers_check(self):
        """DOC.PDF (uppercase) should trigger PDF magic byte check."""
        with pytest.raises(ValueError, match="PDF signature"):
            validate_upload(b"NOT-A-PDF", "DOC.PDF")

    def test_mixed_case_extension_triggers_check(self):
        with pytest.raises(ValueError, match="PDF signature"):
            validate_upload(b"NOT-A-PDF", "Report.Pdf")


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    def test_max_upload_bytes(self):
        assert MAX_UPLOAD_BYTES == 100 * 1024 * 1024

    def test_pdf_magic(self):
        assert PDF_MAGIC == b"%PDF-"

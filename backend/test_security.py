"""
Security regression tests.

Each test encodes an attack that the API previously allowed. Run with:
    python -m pytest backend/test_security.py -q
or standalone:
    python backend/test_security.py
"""

import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import HTTPException

import security


def expect_http_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except HTTPException as exc:
        return exc
    raise AssertionError(f"{fn.__name__} should have rejected the input")


# ---------------------------------------------------------------------------
# Token validation / path traversal
# ---------------------------------------------------------------------------

def test_valid_uuid_token_accepted():
    token = "bc52eabd-22f2-4c4d-90f8-359a15edc47a"
    assert security.validate_token(token) == token


def test_path_traversal_token_rejected():
    # This token is interpolated into recordings/{token}.webm.
    for bad in [
        "../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "/absolute/path",
        "abc",
        "",
        "bc52eabd-22f2-4c4d-90f8-359a15edc47a/../../x",
        "'; DROP TABLE candidates;--",
    ]:
        expect_http_error(security.validate_token, bad)


def test_safe_filename_strips_directories():
    assert security.safe_filename("../../evil.pdf") == "evil.pdf"
    assert security.safe_filename("/etc/passwd") == "passwd"
    assert security.safe_filename("a\\b\\c.docx") == "c.docx"
    assert "/" not in security.safe_filename("x/y/z.pdf")


# ---------------------------------------------------------------------------
# HR authentication
# ---------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, headers=None, ip="1.2.3.4"):
        self.headers = headers or {}

        class _C:
            host = ip

        self.client = _C()


def test_hr_endpoint_denied_without_key():
    os.environ["HR_API_KEY"] = "super-secret"
    exc = expect_http_error(security.require_hr_auth, FakeRequest())
    assert exc.status_code == 401


def test_hr_endpoint_denied_with_wrong_key():
    os.environ["HR_API_KEY"] = "super-secret"
    exc = expect_http_error(
        security.require_hr_auth, FakeRequest({"x-hr-key": "guess"})
    )
    assert exc.status_code == 401


def test_hr_endpoint_allows_correct_key():
    os.environ["HR_API_KEY"] = "super-secret"
    security.require_hr_auth(FakeRequest({"x-hr-key": "super-secret"}))
    security.require_hr_auth(FakeRequest({"authorization": "Bearer super-secret"}))


def test_unconfigured_hr_key_fails_closed():
    # A deployment that forgets to set the key must NOT serve PII openly.
    os.environ.pop("HR_API_KEY", None)
    exc = expect_http_error(security.require_hr_auth, FakeRequest())
    assert exc.status_code == 503
    os.environ["HR_API_KEY"] = "super-secret"


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def test_executable_upload_rejected():
    for name in ["payload.exe", "shell.sh", "evil.php", "run.bat"]:
        expect_http_error(security.validate_resume_upload, name, b"data")


def test_oversized_resume_rejected():
    huge = b"x" * (security.MAX_RESUME_BYTES + 1)
    exc = expect_http_error(security.validate_resume_upload, "cv.pdf", huge)
    assert exc.status_code == 413


def test_empty_resume_rejected():
    expect_http_error(security.validate_resume_upload, "cv.pdf", b"")


def test_normal_resume_accepted():
    security.validate_resume_upload("Candidate_CV.pdf", b"%PDF-1.4 fake")


def test_zip_slip_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../../../tmp/pwned.pdf", "x")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        expect_http_error(security.safe_archive_members, z)


def test_zip_bomb_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # Highly compressible payload that inflates past the cap.
        z.writestr("big.pdf", "0" * (security.MAX_ARCHIVE_UNCOMPRESSED_BYTES + 1))
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        exc = expect_http_error(security.safe_archive_members, z)
        assert exc.status_code == 413


def test_benign_zip_accepted():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.pdf", "x")
        z.writestr("b.docx", "y")
        z.writestr("__MACOSX/junk", "z")
        z.writestr("notes.exe", "ignored")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        members = security.safe_archive_members(z)
    names = sorted(m.filename for m in members)
    assert names == ["a.pdf", "b.docx"], names


# ---------------------------------------------------------------------------
# Proctoring payload sanitisation
# ---------------------------------------------------------------------------

def test_unknown_event_types_dropped():
    events = [
        {"type": "TAB_SWITCH", "timestamp": "2026-01-01T00:00:00Z"},
        {"type": "<script>alert(1)</script>", "timestamp": "x"},
        {"type": "ARBITRARY_JUNK", "timestamp": "x"},
        "not-a-dict",
    ]
    clean = security.sanitize_proctoring_events(events)
    assert len(clean) == 1
    assert clean[0]["type"] == "TAB_SWITCH"


def test_event_fields_are_length_capped():
    clean = security.sanitize_proctoring_events(
        [{"type": "TAB_SWITCH", "timestamp": "T" * 5000, "details": "D" * 5000}]
    )
    assert len(clean[0]["timestamp"]) <= 64
    assert len(clean[0]["details"]) <= 256


def test_event_flood_is_bounded():
    flood = [{"type": "TAB_SWITCH", "timestamp": "t"}] * 20000
    assert len(security.sanitize_proctoring_events(flood)) <= security.MAX_PROCTORING_EVENTS


def test_non_list_events_ignored():
    assert security.sanitize_proctoring_events("nope") == []
    assert security.sanitize_proctoring_events(None) == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limiter_blocks_burst():
    limiter = security.RateLimiter()
    for _ in range(5):
        limiter.check("ip-a", limit=5, window_seconds=60)
    try:
        limiter.check("ip-a", limit=5, window_seconds=60)
        raise AssertionError("6th request should have been rate limited")
    except HTTPException as exc:
        assert exc.status_code == 429


def test_rate_limiter_isolates_clients():
    limiter = security.RateLimiter()
    for _ in range(5):
        limiter.check("ip-a", limit=5, window_seconds=60)
    # A different client must not be affected by another's burst.
    limiter.check("ip-b", limit=5, window_seconds=60)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def test_cors_is_not_wildcard():
    os.environ.pop("ALLOWED_ORIGINS", None)
    origins = security.allowed_origins()
    assert "*" not in origins, "wildcard CORS exposes candidate PII"
    assert all(o.startswith("http") for o in origins)


def test_cors_respects_configuration():
    os.environ["ALLOWED_ORIGINS"] = "https://hr.example.com, https://admin.example.com"
    assert security.allowed_origins() == [
        "https://hr.example.com",
        "https://admin.example.com",
    ]
    os.environ.pop("ALLOWED_ORIGINS")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    print("\nbackend security\n")
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed\n")
    sys.exit(1 if failures else 0)

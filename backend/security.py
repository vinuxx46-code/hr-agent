"""
Security primitives for the HR agent API.

Centralises the controls the interview endpoints depend on:
  * HR authentication (shared secret)
  * strict token format validation (blocks path traversal)
  * upload limits and archive-bomb protection
  * proctoring payload validation
  * lightweight in-process rate limiting

Everything is configured through environment variables so no secret is ever
committed to the repository.
"""

import os
import re
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Interview tokens are UUID4 values generated server-side. Anything else is
# rejected outright - this is what stops "../../etc/passwd" reaching the
# filesystem through the {token} path parameter.
TOKEN_RE = re.compile(
    r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-"
    r"[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
)

ALLOWED_RESUME_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".png", ".jpg", ".jpeg"}

MAX_RESUME_BYTES = int(os.getenv("MAX_RESUME_MB", "10")) * 1024 * 1024
MAX_ARCHIVE_BYTES = int(os.getenv("MAX_ARCHIVE_MB", "50")) * 1024 * 1024
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_MB", "500")) * 1024 * 1024

# Guards against zip bombs: a small archive that inflates to gigabytes.
MAX_ARCHIVE_UNCOMPRESSED_BYTES = (
    int(os.getenv("MAX_ARCHIVE_UNCOMPRESSED_MB", "200")) * 1024 * 1024
)
MAX_ARCHIVE_ENTRIES = int(os.getenv("MAX_ARCHIVE_ENTRIES", "200"))

MAX_PROCTORING_EVENTS = 5000
ALLOWED_PROCTORING_TYPES = {
    "FACE_NOT_DETECTED", "MULTIPLE_FACES", "FACE_OUT_OF_FRAME", "TAB_SWITCH",
    "WINDOW_BLUR", "BACKGROUND_VOICE", "EXTRA_HANDS", "EYES_WANDERING",
    "HEAD_TURNED", "FORBIDDEN_OBJECT_DETECTED", "KEYBOARD_TYPING_DETECTED",
    "360_SCAN_ADDITIONAL_PERSON_DETECTED", "COPY_PASTE_ATTEMPT",
    "FULLSCREEN_EXIT", "DEVTOOLS_OPEN", "MIC_MUTED", "SECOND_SCREEN_DETECTED",
}


# --------------------------------------------------------------------------
# HR authentication
# --------------------------------------------------------------------------

def get_hr_api_key() -> str:
    """Shared secret protecting HR-only endpoints. Empty means unset."""
    return os.getenv("HR_API_KEY", "").strip()


def hr_auth_enabled() -> bool:
    return bool(get_hr_api_key())


def require_hr_auth(request: Request) -> None:
    """
    Guard HR endpoints that expose candidate PII.

    Set HR_API_KEY in the environment and send it as `X-HR-Key`. When the
    variable is unset the API refuses to serve PII at all rather than silently
    running wide open - a misconfigured deployment fails closed.
    """
    expected = get_hr_api_key()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "HR API is not configured. Set HR_API_KEY in the server "
                "environment to enable HR endpoints."
            ),
        )

    provided = request.headers.get("x-hr-key", "")
    if not provided:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()

    # Constant-time comparison avoids leaking the key through timing.
    import hmac
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing HR credentials.",
        )


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def validate_token(token: str) -> str:
    """Reject any token that is not a well-formed UUID."""
    if not token or not TOKEN_RE.match(token):
        raise HTTPException(status_code=400, detail="Malformed interview token.")
    return token


def safe_filename(filename: str) -> str:
    """Strip directory components and dangerous characters from an upload name."""
    if not filename:
        return "upload"
    name = os.path.basename(filename.replace("\\", "/")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    name = name.lstrip(".") or "upload"
    return name[:120]


def validate_resume_upload(filename: str, content: bytes) -> None:
    """Enforce extension allow-list and size cap on a single resume."""
    name = safe_filename(filename).lower()
    ext = os.path.splitext(name)[1]
    if ext not in ALLOWED_RESUME_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: "
                   + ", ".join(sorted(ALLOWED_RESUME_EXTENSIONS)),
        )
    if len(content) > MAX_RESUME_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Resume exceeds the {MAX_RESUME_BYTES // (1024 * 1024)}MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")


def validate_archive(content: bytes) -> None:
    """Size-check a zip upload before it is opened."""
    if len(content) > MAX_ARCHIVE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Archive exceeds the {MAX_ARCHIVE_BYTES // (1024 * 1024)}MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded archive is empty.")


def safe_archive_members(zf) -> list:
    """
    Return the archive entries that are safe to extract.

    Blocks three classic archive attacks:
      * zip-slip   - entries containing .. or absolute paths
      * zip bomb   - huge total uncompressed size
      * entry flood - excessive number of members
    """
    total_uncompressed = 0
    members = []

    for info in zf.infolist():
        name = info.filename
        if name.endswith("/") or name.startswith("__MACOSX"):
            continue

        normalised = os.path.normpath(name.replace("\\", "/"))
        if normalised.startswith("..") or os.path.isabs(normalised) or ".." in normalised.split("/"):
            raise HTTPException(
                status_code=400,
                detail="Archive contains unsafe path entries and was rejected.",
            )

        total_uncompressed += info.file_size
        if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Archive expands beyond the permitted size and was rejected.",
            )

        ext = os.path.splitext(normalised.lower())[1]
        if ext in ALLOWED_RESUME_EXTENSIONS:
            members.append(info)

        if len(members) > MAX_ARCHIVE_ENTRIES:
            raise HTTPException(
                status_code=413,
                detail=f"Archive contains more than {MAX_ARCHIVE_ENTRIES} resumes.",
            )

    return members


def validate_video_upload(content: bytes) -> None:
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Recording exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)}MB limit.",
        )


def sanitize_proctoring_events(events) -> list:
    """
    Normalise a client-supplied proctoring log.

    The browser is untrusted, so only known event types are stored and every
    field is length-capped to keep the JSON database from being used as
    arbitrary storage.
    """
    if not isinstance(events, list):
        return []

    clean = []
    for event in events[:MAX_PROCTORING_EVENTS]:
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", ""))[:64]
        if etype not in ALLOWED_PROCTORING_TYPES:
            continue
        entry = {"type": etype, "timestamp": str(event.get("timestamp", ""))[:64]}
        if "details" in event:
            entry["details"] = str(event.get("details"))[:256]
        clean.append(entry)
    return clean


def sanitize_text(value, limit: int = 20000) -> str:
    """Coerce untrusted input to a bounded string."""
    if value is None:
        return ""
    return str(value)[:limit]


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

class RateLimiter:
    """
    Small in-process sliding-window limiter.

    Sufficient for a single-instance deployment; swap for Redis if the API is
    ever run with multiple workers.
    """

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> None:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = int(bucket[0] + window_seconds - now) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please slow down.",
                    headers={"Retry-After": str(max(retry, 1))},
                )
            bucket.append(now)


rate_limiter = RateLimiter()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, bucket: str, limit: int, window_seconds: int = 60) -> None:
    rate_limiter.check(f"{bucket}:{client_ip(request)}", limit, window_seconds)


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------

def allowed_origins() -> list:
    """
    Explicit origin allow-list.

    ALLOWED_ORIGINS is a comma-separated list. Defaults to local dev hosts so a
    fresh checkout still works without shipping `allow_origins=["*"]`.
    """
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    if configured:
        return [o.strip() for o in configured.split(",") if o.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def allowed_origin_regex() -> str:
    """
    Permit hosted sandbox preview domains without opening CORS to the world.
    Override with ALLOWED_ORIGIN_REGEX when deploying elsewhere.
    """
    return os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.e2b\.app")

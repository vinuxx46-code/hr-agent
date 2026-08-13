"""
Integration tests that exercise the real FastAPI app.

Heavy third-party AI/parsing dependencies are stubbed so the security
behaviour of the actual routes can be verified without network access or the
full ML stack. Run:
    python backend/test_api_integration.py
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# --------------------------------------------------------------------------
# Stub optional heavy dependencies before importing main
# --------------------------------------------------------------------------

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


class _FakeClient:
    def __init__(self, *a, **k):
        self.models = types.SimpleNamespace(generate_content=lambda **kw: types.SimpleNamespace(text="{}"))


_google = _stub("google")
_stub("google.genai", Client=_FakeClient, types=types.SimpleNamespace())
_google.genai = sys.modules["google.genai"]

_stub("PyPDF2", PdfReader=lambda *a, **k: types.SimpleNamespace(pages=[]))
_stub("dotenv", load_dotenv=lambda *a, **k: None)
_stub("knowledge_base", get_knowledge_base=lambda *a, **k: {})
_stub(
    "requirement_engine",
    analyze_resume_against_requirements=lambda *a, **k: {},
    get_requirements=lambda *a, **k: {},
)

os.environ["HR_API_KEY"] = "test-hr-key"
os.chdir(HERE)

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app, raise_server_exceptions=False)

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} {detail}")


print("\napi integration (real routes)\n")

# --------------------------------------------------------------------------
# HR endpoints must require authentication
# --------------------------------------------------------------------------

for path in ["/api/hr/candidates", "/api/hr/dashboard-data", "/api/candidates"]:
    r = client.get(path)
    check(f"{path} rejects unauthenticated access", r.status_code == 401, f"got {r.status_code}")

    r_ok = client.get(path, headers={"X-HR-Key": "test-hr-key"})
    check(f"{path} allows authenticated access", r_ok.status_code == 200, f"got {r_ok.status_code}")

    r_bad = client.get(path, headers={"X-HR-Key": "wrong"})
    check(f"{path} rejects a wrong key", r_bad.status_code == 401, f"got {r_bad.status_code}")

# Verify PII is not leaked in the unauthenticated response body.
body = client.get("/api/hr/candidates").text.lower()
check("no candidate email leaked when unauthorised", "@gmail.com" not in body and "@example.com" not in body)

# --------------------------------------------------------------------------
# Path traversal on token routes
# --------------------------------------------------------------------------

r = client.get("/api/validate-token/..%2F..%2Fetc%2Fpasswd")
check("traversal token rejected on validate-token", r.status_code in (400, 404), f"got {r.status_code}")

r = client.get("/api/validate-token/not-a-uuid")
check("malformed token rejected", r.status_code == 400, f"got {r.status_code}")

r = client.get("/api/check-360-verification/....//....//etc")
check("traversal rejected on check-360", r.status_code in (400, 404), f"got {r.status_code}")

# --------------------------------------------------------------------------
# Security headers + CORS
# --------------------------------------------------------------------------

r = client.get("/api/validate-token/bc52eabd-22f2-4c4d-90f8-359a15edc47a")
check("X-Content-Type-Options header present", r.headers.get("x-content-type-options") == "nosniff")
check("Referrer-Policy header present", r.headers.get("referrer-policy") == "no-referrer")
check("PII response is not cacheable", r.headers.get("cache-control") == "no-store")

r = client.get(
    "/api/hr/candidates",
    headers={"Origin": "https://evil.example.com", "X-HR-Key": "test-hr-key"},
)
acao = r.headers.get("access-control-allow-origin")
check("untrusted origin not granted CORS", acao != "*" and acao != "https://evil.example.com", f"got {acao}")

# --------------------------------------------------------------------------
# Proctoring log injection
# --------------------------------------------------------------------------

r = client.post(
    "/api/log-proctoring-event/bc52eabd-22f2-4c4d-90f8-359a15edc47a",
    json={"type": "ARBITRARY_INJECTED_TYPE", "timestamp": "x" * 9000},
)
check(
    "unknown proctoring event type is not stored",
    r.status_code in (200, 400) and r.json().get("success") is False,
    f"got {r.status_code} {r.text[:120]}",
)

# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

r = client.post(
    "/api/upload-resume",
    files={"resume": ("payload.exe", b"MZ\x90\x00binary", "application/octet-stream")},
)
check("executable upload rejected", r.status_code == 400, f"got {r.status_code}")

r = client.post(
    "/api/bulk-upload",
    files={"file": ("resumes.zip", b"PK\x03\x04notreallyazip", "application/zip")},
)
check("bulk upload requires HR auth", r.status_code == 401, f"got {r.status_code}")

# --------------------------------------------------------------------------
# Answer payload bounds
# --------------------------------------------------------------------------

r = client.post(
    "/api/interview/answer",
    json={
        "sessionId": "s1",
        "questionIndex": 0,
        "answer": "x" * 50000,  # exceeds the 20k cap
        "proctoringEvents": [],
    },
)
check("oversized answer rejected by validation", r.status_code == 422, f"got {r.status_code}")

r = client.post(
    "/api/interview/answer",
    json={"sessionId": "s1", "questionIndex": -5, "answer": "hi", "proctoringEvents": []},
)
check("negative question index rejected", r.status_code == 422, f"got {r.status_code}")

print(f"\n{PASSED}/{PASSED + FAILED} passed\n")
sys.exit(1 if FAILED else 0)

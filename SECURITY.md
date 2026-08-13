# Security Notes

This document records the controls protecting candidate data and how to
configure them.

## Setup

1. Copy the environment template and set a strong HR key:

   ```bash
   cp backend/.env.example backend/.env
   python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste into HR_API_KEY
   ```

2. Start the backend, then open the HR dashboard and enter the same key in the
   **HR access key** field. It is held in `sessionStorage` and sent as the
   `X-HR-Key` header — it is never bundled into the frontend build.

> **`HR_API_KEY` is mandatory.** If it is unset, every HR endpoint returns
> `503` and no candidate data is served. The API fails closed rather than
> silently running wide open.

## Controls

| Area | Control |
|---|---|
| HR endpoints | Shared-secret auth (`X-HR-Key` / `Bearer`), constant-time compare |
| CORS | Explicit origin allow-list; `allow_origins=["*"]` removed |
| Tokens | Strict UUID validation on every `{token}` route |
| Uploads | Extension allow-list, size caps, filename sanitisation |
| Archives | Zip-slip, zip-bomb and entry-flood protection |
| Proctoring logs | Event-type allow-list, field length caps, count caps |
| Answers | `max_length` bounds before text reaches the AI prompt |
| Rate limiting | Per-IP sliding window on auth, upload and logging routes |
| Headers | `nosniff`, `SAMEORIGIN`, `no-referrer`, `no-store` on PII responses |

## Issues fixed

**Unauthenticated PII disclosure.** `/api/hr/candidates`,
`/api/hr/dashboard-data` and `/api/candidates` returned every candidate's name,
email, resume path and evaluation to any anonymous caller. Combined with
`allow_origins=["*"]`, any website could read the entire candidate database
from a visitor's browser. All three now require authentication and CORS is
restricted.

**Path traversal.** `/api/upload-interview-data/{token}` interpolated the token
straight into `recordings/{token}.webm`, so `../../` escaped the recordings
directory and overwrote arbitrary files. Tokens are now validated as UUIDs
before any filesystem use.

**Validation bypass via broad `except`.** `/api/upload-resume` wrapped its body
in a `try` that fell back to the offline engine on *any* exception, swallowing
the upload rejection and returning `200` for a disallowed `.exe`. Validation now
runs before the `try`. This was caught by `test_api_integration.py`.

**Archive attacks.** Bulk upload opened any zip and read every member. A zip
bomb could exhaust memory and a zip-slip entry could write outside the working
directory. Both are now rejected.

**Untrusted proctoring data.** Client-supplied events were persisted verbatim,
letting anyone with a token write unbounded arbitrary JSON into the database.
Events are filtered against an allow-list and length-capped.

**Committed PII.** `hr_database.json` (120 real candidate email addresses),
named resume PDFs and two interview recordings were tracked in Git. They have
been removed from the index and added to `.gitignore`.

> **Note:** these files remain in Git *history*. Removing them entirely
> requires a history rewrite (`git filter-repo`) and a force-push coordinated
> with everyone who has a clone. Treat the exposed addresses as disclosed and
> rotate any credentials that were in `.env` files.

**Hardcoded values.** A personal Gmail address was the fallback HR recipient,
and the dashboard called `http://localhost:8000` directly — unreachable from a
remote browser. Both now use configuration and relative URLs.

## Tests

```bash
python backend/test_security.py           # 22 unit tests
python backend/test_api_integration.py    # 22 tests against real routes
node frontend/src/__tests__/transcript.test.mjs   # 11 voice tests
```

## Remaining hardening

Not addressed here, worth planning:

- HR auth is a single shared secret. Real user accounts with per-user sessions
  and an audit trail would be better for multi-recruiter teams.
- Rate limiting is in-process, so it resets on restart and does not span
  multiple workers. Move to Redis before scaling out.
- The JSON file database is not concurrency-safe; two simultaneous writes can
  lose data. A real database is the durable fix.
- Interview recordings under `recordings/` are served without authentication by
  the static mount; anyone who guesses a token UUID can fetch a video.

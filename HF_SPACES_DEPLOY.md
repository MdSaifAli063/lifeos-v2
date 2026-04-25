# Deploy AetherMind to Hugging Face Spaces

This guide helps you complete the final submission requirement: hosting your OpenEnv-compliant app on Hugging Face Spaces.

## 1) Create Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Suggested settings:
   - **SDK**: Docker (recommended for this repo)
   - **Visibility**: Public
   - **Hardware**: CPU Basic (upgrade if needed)

## 2) Push Your Repository

Use GitHub integration or push directly to the Space git remote.

If using Docker Space, this repo already includes:
- `Dockerfile`
- `docker-compose.yml` (local use)
- Flask app entry (`app/app.py`)

## 3) Runtime Expectations

- App serves at `/` and loads `frontend/frontend.html` via Flask.
- Core APIs exposed:
  - `/resolve`
  - `/emotion`
  - `/rewrite`
  - `/mediate`
  - `/history`
  - `/api/run-episode`
  - `/api/episodes`
  - `/api/episode/latest`

## 4) Post-Deploy Validation

After Space build succeeds:
1. Open your Space URL.
2. Confirm dashboard loads.
3. Test one action from each core module:
   - resolver
   - emotion
   - rewriter
   - mediation
4. Confirm `/health` shows status response.

## 5) Final Submission Updates

Paste Space URL in:
- `README.md` under Demo section
- `SUBMISSION_CHECKLIST.md` at hosted Space line
- `JUDGING_READINESS.md` (optional, for internal tracking)


# Data Analyst Agent

A FastAPI application that exposes a single endpoint `/api/` to accept a `questions.txt` file and optional attachments (CSV, JSON, images). The service uses local Python tooling (pandas, duckdb, matplotlib, etc.) and optionally the OpenAI API to plan or parse tasks.

## Features

- Accepts `questions.txt` and arbitrary attachments
- Parses tasks and attempts to satisfy questions using available data or by scraping provided URLs
- Produces JSON responses in the exact formats requested by the prompt (JSON array of strings or JSON object as appropriate)
- Produces base64-encoded images (data URI) for plots and ensures they are reasonably compressed (tries to keep under 100kB when asked)
- Uses OpenAI as an optional helper (must set `OPENAI_API_KEY` in env)

## Quickstart

1. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables (or copy `.env.example` -> `.env`):

```
OPENAI_API_KEY=sk-...
PORT=8000
```

3. Run the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Example curl (from `examples/curl_example.sh`):

```bash
bash examples/curl_example.sh
```

## Notes and Known Limitations

- The project aims to be general-purpose but cannot guarantee successful answers for *every* secret test. It will try to load CSVs, JSON, read HTML tables, and scrape data if a URL is present.
- The OpenAI API is optional — if `OPENAI_API_KEY` is not provided the system falls back to deterministic analysis.
- The service attempts to keep plotted images under 100 KB by automatically downsampling or converting to WEBP when requested.

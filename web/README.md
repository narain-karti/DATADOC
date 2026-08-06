# DATADOC web dashboard

This directory contains the optional React frontend for DATADOC. The backend remains the source of truth: the UI calls the same profile, plan, fit, preview, and export endpoints used by the local dashboard.

## Development

From this directory:

```bash
npm ci
npm run build
```

The production build is written to `web/dist/` and is served by the FastAPI UI server when present.

For Vite development against a separately running DATADOC API, set the API base explicitly:

```bash
$env:VITE_DATADOC_API_URL = "http://127.0.0.1:8000/api"
npm run dev
```

The frontend intentionally keeps AI optional. It displays the safety warning that generated code is never executed and always sends the local session header with API requests.

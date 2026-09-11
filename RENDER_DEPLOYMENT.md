# Render deployment — PIRI

## Before deployment

Make sure the repository is pushed to GitHub and contains `render.yaml` at the repository root.

## Render steps

1. Sign in to Render.
2. Select **New → Blueprint**.
3. Connect GitHub and select the PIRI repository.
4. Review the two resources from `render.yaml`:
   - `piri-ai-monitoring`
   - `piri-postgres`
5. Deploy.
6. Watch the web-service deploy logs.
7. When the service is live, open its `onrender.com` URL.
8. Test:
   - `/health`
   - `/docs`
   - `/`

Expected health response includes `status: ok` and `model_loaded: true` after bootstrap completes.

## If deployment fails

Open the web service's deploy logs and look for the first Python traceback. Common issues are:

- dependency installation failure
- PostgreSQL connection/configuration failure
- model training failure
- insufficient labeled records

Do not add PAIMANA credentials until the synthetic deployment is working.

## Free-tier warning

Render Free Web Services can sleep when idle. Free Render Postgres currently has a 1 GB limit and expires 30 days after creation. This is acceptable for an SIH demonstration but not for permanent production storage.

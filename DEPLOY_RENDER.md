# Deploy LawGPT on Render

This project is configured to deploy as a Streamlit web service on Render.

## Files Added

- `render.yaml`: Render Blueprint config for build/start commands and env vars.

## Before Deploying

1. Push this project to a GitHub repository.
2. Make sure `my_vector_store/` is committed (contains `index.faiss` and `index.pkl`).
3. Keep your `.env` out of Git (already expected). You will set secrets in Render.

## Deploy Steps

1. Open Render dashboard: https://dashboard.render.com/
2. Click `New` -> `Blueprint`.
3. Connect your GitHub repo and select this repository.
4. Render will detect `render.yaml` automatically.
5. In environment variables, set:
   - `GROQ_API_KEY` = your Groq key
6. Deploy.

## Runtime Notes

- Streamlit runs with:
  - `--server.port $PORT`
  - `--server.address 0.0.0.0`
  - `--server.headless true`
- OCR for image uploads requires `tesseract-ocr`, installed via build command.

## If Build Fails on System Packages

If Render blocks `apt-get` in your selected plan/runtime, remove OCR dependency temporarily:

1. Remove `pytesseract` usage in `app.py` upload path for images.
2. Keep PDF support (`PyMuPDF`) and redeploy.

## Local Test Command

```bash
streamlit run app.py
```

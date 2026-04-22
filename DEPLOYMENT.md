# Deployment Notes

## Netlify

This project cannot be deployed as a full working app on Netlify Functions in its current form.

Why:
- Netlify's official Functions runtime supports JavaScript, TypeScript, and Go.
- This app is a Python Flask backend.
- The backend also loads TensorFlow, PyTorch, Hugging Face Transformers, and local `.h5` model files, which makes it too heavy for a typical serverless setup.

Official Netlify docs:
- https://docs.netlify.com/build/functions/overview/
- https://docs.netlify.com/build/functions/get-started/

## Recommended setup

Use:
- Netlify for a static frontend only, if you want
- Render or Railway for the Python ML backend

## Fastest working option: Render

This repository now includes:
- `requirements.txt`
- `app.py`
- `render.yaml`

### Render steps

1. Push this project to GitHub.
2. Create a new Web Service in Render.
3. Connect your GitHub repository.
4. Render should detect `render.yaml` automatically.
5. Deploy.

Start command:

```bash
gunicorn app:app
```

## Important note about Hugging Face model download

Your app loads `distilgpt2` with:

```python
AutoTokenizer.from_pretrained("distilgpt2")
AutoModelForCausalLM.from_pretrained("distilgpt2")
```

That means the host must be able to download model files during the first build or first startup unless you package them manually.

## If you still want Netlify

The workable architecture is:
- frontend on Netlify
- backend API on Render/Railway

Then the Netlify frontend calls the deployed backend endpoint for prediction.

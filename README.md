# AI Edit Suite

AI Edit Suite exposes a FastAPI service that wraps a small collection of
computer‑vision utilities.  Using DeepLabV3 segmentation and an inpainting
model it can extract, remove, resize or overlay objects within images.

The code is organised as a Python package so it can be imported or run as a
stand‑alone API.

## Structure

- `app.py` – FastAPI application entry point.
- `ai_edit_suite/` – Python package containing the core logic.
  - `extract.py` – helpers to segment and extract object masks.
  - `resize.py` – resize objects after removing them from the background.
  - `overlap.py` – overlay objects from one image onto another.
  - `segmentation.py` – utilities to load segmentation models.
  - `object_eraser/` – lightweight wrapper around an inpainting model.

## Development

Install the dependencies and start the API locally:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

A `Dockerfile` is provided so the API can also be run inside Docker:

```bash
docker build -t ai-edit-suite .
docker run -p 8000:8000 ai-edit-suite
```

When the server is running you can access the automatic API docs at
`http://localhost:8000/docs`.

### Endpoints

- `POST /extract_single/` – extract a single object class from an uploaded image
  and return a cropped PNG with transparency.
- `POST /extract_multiple/` – extract multiple classes and return a ZIP archive
  containing the cropped instances.
- `POST /remove/` – erase instances of the specified classes using inpainting.
- `POST /resize_single/` – resize the first instance of a class by a scale
  factor.
- `POST /resize_multiple/` – sequentially resize multiple classes.
- `POST /overlap/` – overlay objects from one image onto another image.

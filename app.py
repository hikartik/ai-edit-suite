# app.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from ai_edit_suite.extract import (
    extract_single_instance_from_pil,
    extract_multiple_classes_to_zip_bytes,
)
from ai_edit_suite.resize import (
    resize_single_instance_from_pil,
    resize_multiple_classes_from_pil,
)
from ai_edit_suite.overlap import overlap_classes_from_pil
from ai_edit_suite import segmentation
from ai_edit_suite.object_eraser.predict import init_inpainter, remove_objects_from_pil
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # Load segmentation models
    segmentation.load_segmentation_model(seg_net=0)
    segmentation.load_segmentation_model(seg_net=1)  # if you use FCN
    # Load inpainting model
    init_inpainter()
    # You can log:
    print("Models loaded at startup.")
@app.get("/")
async def root():
    return {"message": "Use POST /extract_single/, /extract_multiple/, or /remove/"}

@app.post("/extract_single/")
async def extract_single_endpoint(
    file: UploadFile = File(...),
    class_name: str = Form(...)
):
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    name = class_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="No class_name provided")
    try:
        out_img = extract_single_instance_from_pil(pil, name)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/extract_multiple/")
async def extract_multiple_endpoint(
    file: UploadFile = File(...),
    classes: str = Form(...)
):
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    names = [c.strip() for c in classes.split(",") if c.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="No classes provided")
    try:
        zip_bytes = extract_multiple_classes_to_zip_bytes(pil, names)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="extracted.zip"'}
    )

@app.post("/remove/")
async def remove_endpoint(
    file: UploadFile = File(...),
    classes: str = Form(...)
):
    """
    Remove (erase) all instances of given classes from the image.
    classes: comma-separated class names, e.g. "person,car"
    """
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    names = [c.strip() for c in classes.split(",") if c.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="No classes provided")
    try:
        out_img = remove_objects_from_pil(pil, names)
    except ValueError as ve:
        # if your removal function raises ValueError for unknown class or similar
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Removal error: {e}")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/resize_single/")
async def resize_single_endpoint(file: UploadFile = File(...),
                                 class_name: str = Form(...),
                                 scale: float = Form(...)):
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")
    try:
        out_img = resize_single_instance_from_pil(pil, class_name, scale)
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, f"Resize error: {e}")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/resize_multiple/")
async def resize_multiple_endpoint(
    file: UploadFile = File(...),
    classes: str = Form(...),
    scales: str = Form(...)
):
    """
    classes: comma-separated class names, e.g. "person,car"
    scales: comma-separated floats matching classes, e.g. "0.5,1.2"
    """
    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    names = [c.strip() for c in classes.split(",") if c.strip()]
    scales_list = [s.strip() for s in scales.split(",") if s.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="No classes provided")
    if len(names) != len(scales_list):
        raise HTTPException(status_code=400, detail="classes and scales must have same number of items")
    try:
        scales_f = [float(s) for s in scales_list]
    except:
        raise HTTPException(status_code=400, detail="Invalid scale values")
    for s in scales_f:
        if s <= 0:
            raise HTTPException(status_code=400, detail="Scale values must be > 0")
    try:
        out_img = resize_multiple_classes_from_pil(pil, names, scales_f)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resize multiple error: {e}")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



@app.post("/overlap/")
async def overlap_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    classes: str = Form(...)
):
    """
    Upload two images and comma-separated classes from image1 to overlay onto image2.
    Returns a single PNG.
    """
    data1 = await file1.read()
    data2 = await file2.read()
    try:
        pil1 = Image.open(io.BytesIO(data1)).convert("RGB")
        pil2 = Image.open(io.BytesIO(data2)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file(s)")
    names = [c.strip() for c in classes.split(",") if c.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="No classes provided")
    try:
        out_img = overlap_classes_from_pil(pil1, pil2, names)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overlap error: {e}")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
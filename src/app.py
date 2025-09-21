import io
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from core import configure_logging, env, settings
from core.tools import (
    draw_comparison_overlay,
    ensure_ndarray_rgb,
    evaluate_detections,
    parse_coco_annotations,
    preds_from_results,
)
from model import AsyncYoloModel

configure_logging(env.log_level, env.log_type)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model once; optional warmup; release on shutdown."""
    model = await AsyncYoloModel.create(
        path=settings.model.path,
        device=settings.model.device,
        imgsz=settings.model.imgsz,
        conf=settings.model.conf,
        warmup=settings.model.warmup,
        concurrent=settings.model.concurrent,
    )
    app.state.model = model

    logger.info(
        "Model initialized (path=%s, device=%s)", settings.model.path, settings.model.device
    )

    try:
        yield
    finally:
        await model.aclose()
        app.state.model = None
        logger.info("Model released.")


app = FastAPI(title="Doc Layout API", version="0.1", lifespan=lifespan)


def _get_model() -> AsyncYoloModel:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")
    return cast("AsyncYoloModel", model)


@app.post("/detect")
async def detect(
    image: Annotated[UploadFile, File(description="Image file (PNG/JPG/WebP)")] = ...,
) -> Response:
    """
    Accepts an image and returns an annotated image (image/png).
    If nothing is detected, returns the original image as PNG.
    """
    # Read & decode uploaded image
    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    # Run model predict
    try:
        det_res_list = await _get_model().predict(pil)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    # Return the original image unchanged as PNG (no detections)
    if not det_res_list:
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    annotated = det_res_list[0].plot(pil=True, line_width=5, font_size=20)
    annotated_image = ensure_ndarray_rgb(annotated)
    out_buf = io.BytesIO()
    annotated_image.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")


@app.post("/evaluate")
async def evaluate(
    image: Annotated[UploadFile, File(description="Image file (PNG/JPG/WebP)")] = ...,
    annotations: Annotated[UploadFile, File(description="COCO annotations JSON")] = ...,
) -> Response:
    """
    Accepts an image and a COCO annotation file. Returns a comparison image (PNG).
    The evaluation metrics are provided in the X-Metrics response header (JSON).
    """
    # 1) parse image
    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    # 2) parse COCO
    try:
        ann_bytes = await annotations.read()
        ref = parse_coco_annotations(ann_bytes)
        gts = ref["boxes"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid COCO annotations: {e}") from e

    # 3) predict
    try:
        results = await _get_model().predict(pil)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    # 4) to common preds
    preds = preds_from_results(results)

    # 5) evaluate
    metrics, matches = evaluate_detections(preds=preds, gts=gts, iou_thr=0.5)

    # 6) draw overlay (GT — green, Pred — red)
    overlay = draw_comparison_overlay(pil, preds=preds, gts=gts, id_to_class=ref["id_to_class"])
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # 7) return PNG + metrics in header
    headers = {"X-Metrics": json.dumps({"metrics": metrics, "counts": matches})}
    return Response(content=png_bytes, media_type="image/png", headers=headers)

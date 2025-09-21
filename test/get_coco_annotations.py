import asyncio
import json
import sys
from typing import Any

import aiofiles
import numpy as np
from PIL import Image

from core import env
from model import AsyncYoloModel


def _xyxy_to_xywh(b: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, b)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def yolo_result_to_arrays(res: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract xyxy [N,4], cls [N], conf [N] from one Ultralytics-like result.
    """
    bx = getattr(res, "boxes", None)
    if bx is None:
        raise ValueError("Unsupported result: no .boxes")

    # coords
    if hasattr(bx, "xyxy"):
        xyxy = bx.xyxy.detach().cpu().numpy()
    elif hasattr(bx, "data"):
        xyxy = bx.data.detach().cpu().numpy()[:, :4]
    else:
        raise ValueError("Unknown boxes structure")

    # class ids
    if getattr(bx, "cls", None) is not None:
        cls = bx.cls.detach().cpu().numpy()
    else:
        cls = np.zeros((xyxy.shape[0],), dtype=np.float32)

    # confidences
    if getattr(bx, "conf", None) is not None:
        conf = bx.conf.detach().cpu().numpy()
    else:
        conf = np.ones((xyxy.shape[0],), dtype=np.float32)

    return xyxy, cls, conf


def yolo_to_coco_annotations(
    results_any: Any,
    image_id: int = 1,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Convert YOLO predictions for ONE image into a minimal COCO dict.
    """
    res0 = results_any[0] if isinstance(results_any, (list, tuple)) else results_any
    xyxy, cls, conf = yolo_result_to_arrays(res0)

    # Try to infer image size
    infer_width = infer_height = None
    if hasattr(res0, "orig_shape"):
        infer_height, infer_width = map(int, res0.orig_shape)

    anns: list[dict[str, Any]] = []
    cats_set: set[int] = set()
    for i in range(xyxy.shape[0]):
        x, y, w, h = _xyxy_to_xywh(xyxy[i])
        cid = int(cls[i])
        cats_set.add(cid)
        anns.append(
            {
                "id": i + 1,
                "image_id": image_id,
                "category_id": cid,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
                "score": float(conf[i]),
            }
        )

    # categories
    categories = []
    if id_to_name is None:
        names = None
        if hasattr(res0, "names") and isinstance(res0.names, (list, dict)):
            names = res0.names
        elif hasattr(getattr(res0, "model", None), "names"):
            names = res0.model.names
        if isinstance(names, dict):
            id_to_name = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            id_to_name = dict(enumerate(names))
        else:
            id_to_name = {}
    categories = [{"id": cid, "name": id_to_name.get(cid, str(cid))} for cid in sorted(cats_set)]

    return {
        "images": [
            {
                "id": image_id,
                **(
                    {"width": infer_width, "height": infer_height}
                    if infer_width and infer_height
                    else {}
                ),
            }
        ],
        "categories": categories,
        "annotations": anns,
    }


async def main() -> None:
    if len(sys.argv) != 3:  # noqa: PLR2004
        print("Usage: python get_coco_annotations.py <input_image> <output_json>")
        sys.exit(2)

    in_file = sys.argv[1]
    out_file = sys.argv[2]

    model = await AsyncYoloModel.create(
        path=env.model_path,
        device=env.model_device,
        imgsz=env.model_imgsz,
        conf=env.model_conf,
        warmup=env.model_warmup,
        concurrent=env.model_concurrent,
    )

    try:
        pil = Image.open(in_file).convert("RGB")
        results = await model.predict(pil)
        coco = yolo_to_coco_annotations(results, image_id=42)

        async with aiofiles.open(out_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(coco, ensure_ascii=False, indent=2))

        print(f"Saved COCO predictions to {out_file}")
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())

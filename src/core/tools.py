import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ensure_ndarray_rgb(img: np.ndarray) -> Image.Image:
    arr = img
    if hasattr(arr, "ndim"):
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):  # noqa: PLR2004
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 3:  # noqa: PLR2004
            rgb = arr[:, :, ::-1]
            return Image.fromarray(rgb, mode="RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:  # noqa: PLR2004
            rgba = arr[:, :, [2, 1, 0, 3]]
            return Image.fromarray(rgba, mode="RGBA").convert("RGB")
    raise ValueError(f"Unsupported ndarray shape or type: {type(img)}")


def _xywh_to_xyxy(b: list[float]) -> list[float]:
    x, y, w, h = map(float, b[:4])
    return [x, y, x + w, y + h]


def parse_coco_annotations(ann_bytes: bytes) -> dict[str, Any]:
    """
    {
      "categories":[{"id":1,"name":"text"},...],
      "annotations":[{"bbox":[x,y,w,h],"category_id":1}, ...]
    }
    -> {"boxes":[{"bbox":[x1,y1,x2,y2], "cls_id":int},...], "class_to_id":..., "id_to_class":...}
    """
    data = json.loads(ann_bytes.decode("utf-8"))
    if not isinstance(data.get("categories"), list) or not isinstance(
        data.get("annotations"), list
    ):
        raise TypeError("COCO must contain 'categories' and 'annotations' lists")

    id_to_class = {int(c["id"]): str(c["name"]) for c in data["categories"]}
    class_to_id = {v: k for k, v in id_to_class.items()}

    boxes = []
    for a in data["annotations"]:
        if "bbox" not in a or "category_id" not in a:
            continue
        bbox = _xywh_to_xyxy(a["bbox"])
        cls_id = int(a["category_id"])
        boxes.append({"bbox": bbox, "cls_id": cls_id})

    return {"boxes": boxes, "class_to_id": class_to_id, "id_to_class": id_to_class}


def _boxes_from_result_single(res: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bx = getattr(res, "boxes", None)
    if bx is None:
        if isinstance(res, dict):
            xyxy = np.asarray(res["xyxy"], dtype=float)
            cls = np.asarray(res.get("cls", np.zeros(len(xyxy))), dtype=float)
            conf = np.asarray(res.get("conf", np.ones(len(xyxy))), dtype=float)
            return xyxy, cls, conf
        raise ValueError("Unsupported result structure: no 'boxes'")

    # xyxy
    if hasattr(bx, "xyxy"):
        xyxy = np.asarray(bx.xyxy.cpu().numpy(), dtype=float)
    elif hasattr(bx, "data"):
        data = np.asarray(bx.data.cpu().numpy(), dtype=float)
        xyxy = data[:, :4]
    else:
        raise ValueError("Unknown boxes structure")

    # cls
    if hasattr(bx, "cls") and bx.cls is not None:
        cls = np.asarray(bx.cls.cpu().numpy(), dtype=float)
    else:
        cls = np.zeros((xyxy.shape[0],), dtype=float)

    # conf
    if hasattr(bx, "conf") and bx.conf is not None:
        conf = np.asarray(bx.conf.cpu().numpy(), dtype=float)
    else:
        conf = np.ones((xyxy.shape[0],), dtype=float)

    return xyxy, cls, conf


def preds_from_results(results_any: Any) -> list[dict[str, Any]]:
    """
    [{"bbox":[x1,y1,x2,y2], "cls_id":int, "conf":float}, ...]
    """
    res0 = results_any[0] if isinstance(results_any, (list, tuple)) else results_any
    xyxy, cls, conf = _boxes_from_result_single(res0)
    return [
        {"bbox": xyxy[i].tolist(), "cls_id": int(cls[i]), "conf": float(conf[i])}
        for i in range(xyxy.shape[0])
    ]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def evaluate_detections(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    iou_thr: float = 0.5,
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Greedy one-to-one matching per class, sorted by confidence.
    Returns (metrics_dict, counts_dict).
    """

    def group_by_class(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        groups: dict[int, list[dict[str, Any]]] = {}
        for it in items:
            cls_id = int(it["cls_id"])
            groups.setdefault(cls_id, []).append(it)
        return groups

    preds_by_class = group_by_class(preds)
    gts_by_class = group_by_class(gts)

    tp = fp = fn = 0
    ious_accum: list[float] = []

    all_scores: list[float] = []
    all_tp_flags: list[int] = []
    total_positives = len(gts)

    for cls_id in set(preds_by_class) | set(gts_by_class):
        sorted_preds = sorted(preds_by_class.get(cls_id, []), key=lambda x: -x["conf"])
        gt_list = gts_by_class.get(cls_id, [])
        matched = [False] * len(gt_list)

        for pr in sorted_preds:
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_list):
                if matched[j]:
                    continue
                val = _iou(pr["bbox"], gt["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j

            if best_iou >= iou_thr and best_j >= 0:
                matched[best_j] = True
                tp += 1
                ious_accum.append(best_iou)
                all_scores.append(pr["conf"])
                all_tp_flags.append(1)
            else:
                fp += 1
                all_scores.append(pr["conf"])
                all_tp_flags.append(0)

        fn += matched.count(False)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    miou = float(np.mean(ious_accum)) if ious_accum else 0.0

    if all_scores and total_positives > 0:
        order = np.argsort(-np.asarray(all_scores))
        tp_cum = np.cumsum(np.asarray(all_tp_flags)[order])
        fp_cum = np.cumsum(1 - np.asarray(all_tp_flags)[order])
        recalls = tp_cum / total_positives
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            p_at_r = float(np.max(precisions[mask])) if np.any(mask) else 0.0
            ap += p_at_r
        ap /= 11.0
    else:
        ap = 0.0

    metrics = {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "mIoU": round(float(miou), 6),
        "mAP@0.5": round(float(ap), 6),
    }
    counts = {"tp": tp, "fp": fp, "fn": fn}
    return metrics, counts


def _label_text(id_to_class: dict[int, str], cls_id: int, conf: float | None = None) -> str:
    name = id_to_class.get(int(cls_id), str(cls_id))
    return f"{name} {conf:.2f}" if conf is not None else f"{name}"


def draw_comparison_overlay(
    pil: Image.Image,
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    id_to_class: dict[int, str],
) -> Image.Image:
    """
    GT: green boxes, Pred: red boxes
    """
    img = pil.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except (OSError, ImportError):
        # не удалось загрузить встроенный шрифт
        font = None

    # GT green
    for gt in gts:
        x1, y1, x2, y2 = gt["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        if font:
            draw.text(
                (x1 + 2, y1 + 2),
                _label_text(id_to_class, gt["cls_id"]),
                fill=(0, 200, 0),
                font=font,
            )

    # predictions red
    for pr in preds:
        x1, y1, x2, y2 = pr["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=3)
        if font:
            draw.text(
                (x1 + 2, max(0, y1 - 12)),
                _label_text(id_to_class, pr["cls_id"], pr.get("conf")),
                fill=(220, 0, 0),
                font=font,
            )

    return img

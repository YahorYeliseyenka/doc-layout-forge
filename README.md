# Doc Layout API

FastAPI service for document layout detection based on `doclayout_yolo` (YOLOv10).

It exposes two endpoints:

- `POST /detect` — accepts an image and returns an annotated PNG.
- `POST /evaluate` — accepts an image and COCO annotations, returns a PNG overlay (GT green, predictions red) and evaluation metrics in the HTTP header.

---

## Environment (`.env`)

```env
LOG_LEVEL=INFO
LOG_TYPE=TEXT            # TEXT | JSON

MODEL_PATH=models/doclayout_yolo_docstructbench_imgsz1024.pt
MODEL_DEVICE=cpu         # cpu | cuda:<index>  (e.g., cuda:0)
MODEL_IMGSZ=1024
MODEL_CONF=0.20
MODEL_WARMUP=true
MODEL_CONCURRENT=2
```

---

## Build & Run

### Using Makefile

CPU (default if `MODEL_DEVICE=cpu`):

```bash
make build
make run
```

GPU (set in `.env`):

```env
MODEL_DEVICE=cuda:0
```

```bash
make build
make run
```

Useful commands:

```bash
make help
make logs
make stop
make rm
make rebuild
```

### Local development (without Docker)

```bash
uv sync --locked
uv run uvicorn src.app:app --host 0.0.0.0 --port 49494
```

---

## API

Base URL: `http://localhost:49494`
Docs: **/docs** (Swagger) or **/redoc**

### `POST /detect`

- **Request:** multipart/form-data
  - `image`: PNG/JPG/WebP
- **Response:** `200 OK` → `image/png` (annotated image or original if no detections)

Example:

```bash
curl -s -X POST "http://localhost:49494/detect"   -F "image=@test/example/academic.jpg"   --output test/results/out_detect.png
```

---

### `POST /evaluate`

- **Request:** multipart/form-data
  - `image`: PNG/JPG/WebP
  - `annotations`: COCO JSON
- **Response:**
  - `200 OK` → `image/png` (GT green, predictions red)
  - HTTP header `X-Metrics` with JSON:
    ```json
    {
      "metrics": {"precision": 0.91, "recall": 0.88, "f1": 0.895, "mIoU": 0.73, "mAP@0.5": 0.81},
      "counts": {"tp": 42, "fp": 3, "fn": 6}
    }
    ```

Example:

```bash
curl -s -X POST "http://localhost:49494/evaluate"   -F "image=@test/example/ppt.jpg"   -F "annotations=@test/example/ppt.json"   -D test/results/headers.txt   --output test/results/out_eval.png
grep -i '^X-Metrics:' test/results/headers.txt
```

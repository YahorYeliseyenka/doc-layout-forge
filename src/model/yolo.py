import asyncio
from pathlib import Path
from typing import Any, Self

from doclayout_yolo import YOLOv10
from PIL import Image
from pydantic import BaseModel


class AsyncYoloModel(BaseModel):
    path: Path
    device: str
    imgsz: int
    conf: float
    warmup: bool
    concurrent: int

    _model: YOLOv10 | None = None
    _sem: asyncio.Semaphore | None = None

    @classmethod
    async def create(cls, **kwargs: Any) -> Self:
        self = cls(**kwargs)
        await self.init()
        return self

    async def init(self) -> None:
        """Load model and prepare concurrency primitives."""
        self._model = YOLOv10(self.path)
        self._sem = asyncio.Semaphore(self.concurrent)

        if self.warmup:
            dummy = Image.new("RGB", (self.imgsz, self.imgsz), color=(255, 255, 255))
            await asyncio.to_thread(
                self._model.predict,
                dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
            )

    def _require_sem(self) -> asyncio.Semaphore:
        sem = self._sem
        if sem is None:
            raise RuntimeError("AsyncYoloModel is not initialized: semaphore is None")
        return sem

    def _require_model(self) -> YOLOv10:
        model = self._model
        if model is None:
            raise RuntimeError("AsyncYoloModel is not initialized: model is None")
        return model

    async def predict(self, img: Image.Image) -> list[Any]:
        """Run single-image inference in a thread without blocking the event loop."""
        sem = self._require_sem()
        model = self._require_model()
        async with sem:
            return await asyncio.to_thread(
                model.predict,
                img,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
            )

    async def aclose(self) -> None:
        """Release references"""
        self._model = None
        self._sem = None

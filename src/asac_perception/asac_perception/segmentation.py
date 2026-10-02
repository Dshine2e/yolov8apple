"""YOLO instance masks in original RGB coordinates. Never synthesize masks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    bbox: np.ndarray
    confidence: float
    class_id: int
    mask: np.ndarray


class Segmenter:
    def __init__(self, path, classes="apple", device="cpu", image_size=640, cpu_threads=2):
        import torch
        from ultralytics import YOLO

        if not Path(path).is_file():
            raise ValueError(f"Segmentation weights missing: {path}")
        self.model = YOLO(str(path))  # infer task from checkpoint, not filename
        if self.model.task != "segment":
            raise ValueError(f"Expected segmentation checkpoint; got {self.model.task}")
        names = self.model.names
        names = names if isinstance(names, dict) else dict(enumerate(names))
        wanted = {s.strip() for s in classes.split(",") if s.strip()}
        if not wanted or wanted - set(names.values()):
            raise ValueError(f"Configure actual apple class name(s); model.names={names}")
        self.ids = [int(i) for i, name in names.items() if name in wanted]
        self.device, self.image_size = device, image_size
        self.predict(np.zeros((image_size, image_size, 3), np.uint8), 0.5, 0.45, 100)
        if device == "cpu":
            torch.set_num_threads(cpu_threads)

    def predict(self, bgr, confidence, iou, max_detections):
        result = self.model.predict(
            bgr,
            classes=self.ids,
            conf=confidence,
            iou=iou,
            imgsz=self.image_size,
            device=self.device,
            max_det=max_detections,
            retina_masks=True,
            verbose=False,
        )[0]
        if len(result.boxes) == 0:
            return []
        if result.masks is None:
            raise ValueError("Segmentation model returned boxes without instance masks")
        # v8.3.40 retina_masks=True removes letterbox and upsamples before crop.
        masks = result.masks.data.cpu().numpy()
        if masks.shape != (len(result.boxes), *bgr.shape[:2]):
            raise ValueError(f"Original-resolution mask contract violated: {masks.shape}")
        boxes = result.boxes.cpu().numpy()
        instances = []
        for bbox, score, cls, mask in zip(boxes.xyxy, boxes.conf, boxes.cls, masks):
            if not np.all(np.isfinite(bbox)) or not np.isfinite(score):
                raise ValueError("Nonfinite model output")
            instances.append(Instance(bbox.astype(float), float(score), int(cls), mask > 0.5))
        return instances

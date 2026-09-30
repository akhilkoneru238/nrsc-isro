from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ModelName = Literal["yolo", "faster-rcnn"]


class Detection(BaseModel):
    id: int
    x: float                    
    y: float                      
    w: float
    h: float
    cx: float                    
    cy: float
    diameter: float               
    score: float
    label: str = "crater"


class ModelInfo(BaseModel):
    name: ModelName
    display_name: str
    loaded: bool
    weights: str
    available: bool
    detail: Optional[str] = None
    device: str


class Health(BaseModel):
    status: str
    version: str
    device: str
    defaults: Dict[str, float]
    models: List[ModelInfo]


class DetectResponse(BaseModel):
    model: ModelName
    image_width: int
    image_height: int
    conf: float
    iou: float
    imgsz: int
    max_det: int
    count: int
    inference_ms: float
    detections: List[Detection]
    overlay_png: Optional[str] = None

    model_config = {"protected_namespaces": ()}
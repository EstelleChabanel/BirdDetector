# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, DeepCoralDetectionPredictor
from .train import DetectionTrainer, DeepCoralDetectionTrainer
from .val import DetectionValidator, DeepCoralDetectionValidator

__all__ = 'DetectionPredictor', 'DeepCoralDetectionPredictor', 'DetectionTrainer', 'DetectionValidator', 'DeepCoralDetectionValidator', 'DeepCoralDetectionTrainer'

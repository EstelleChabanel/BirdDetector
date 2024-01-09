# Ultralytics YOLO 🚀, AGPL-3.0 license

#from yolo.models.yolo import classify, detect, pose, segment
from yolo.models.yolo import detect

from .model import YOLO

#__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO'
__all__ =  'detect', 'YOLO'

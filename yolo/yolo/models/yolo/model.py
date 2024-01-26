# Ultralytics YOLO 🚀, AGPL-3.0 license

from yolo.engine.model import Model
from yolo.models import yolo  # noqa
#from yolo.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel
from yolo.nn.tasks import DetectionModel, DomainClassifier, MultiDomainClassifier, FeaturesDistance, MultiFeaturesSingleDomainClassifier, UnsupervisedDomainClassifier


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'domainclassifier': {
                'model': DomainClassifier,
                'trainer': yolo.detect.DomainClassifierTrainer,
                'validator': yolo.detect.DomainClassifierValidator,
                'predictor': yolo.detect.DomainClassifierPredictor, },
            'multidomainclassifier': {
                'model': MultiDomainClassifier,
                'trainer': yolo.detect.MultiDomainClassifierTrainer,
                'validator': yolo.detect.DomainClassifierValidator,
                'predictor': yolo.detect.DomainClassifierPredictor, },
            'multifeaturesDC': {
                'model': MultiFeaturesSingleDomainClassifier,
                'trainer': yolo.detect.MultiFeaturesSingleDomainClassifierTrainer,
                'validator': yolo.detect.DomainClassifierValidator,
                'predictor': yolo.detect.DomainClassifierPredictor, },
            'featuresdistance': {
                'model': FeaturesDistance,
                'trainer': yolo.detect.FeaturesDistanceTrainer,
                'validator': yolo.detect.DomainClassifierValidator,
                'predictor': yolo.detect.DomainClassifierPredictor, }, 
                
            'unsuperviseddomainclassifier': {
                'model': UnsupervisedDomainClassifier,
                'trainer': yolo.detect.UnsupervisedDomainClassifierTrainer,
                'validator': yolo.detect.DomainClassifierValidator,
                'predictor': yolo.detect.DomainClassifierPredictor,
            }}
    
    """
    @property
        def task_map(self):
            #Map head to model, trainer, validator, and predictor classes.
            return {
                'classify': {
                    'model': ClassificationModel,
                    'trainer': yolo.classify.ClassificationTrainer,
                    'validator': yolo.classify.ClassificationValidator,
                    'predictor': yolo.classify.ClassificationPredictor, },
                'detect': {
                    'model': DetectionModel,
                    'trainer': yolo.detect.DetectionTrainer,
                    'validator': yolo.detect.DetectionValidator,
                    'predictor': yolo.detect.DetectionPredictor, },
                'segment': {
                    'model': SegmentationModel,
                    'trainer': yolo.segment.SegmentationTrainer,
                    'validator': yolo.segment.SegmentationValidator,
                    'predictor': yolo.segment.SegmentationPredictor, },
                'pose': {
                    'model': PoseModel,
                    'trainer': yolo.pose.PoseTrainer,
                    'validator': yolo.pose.PoseValidator,
                    'predictor': yolo.pose.PosePredictor, }, }
    """
# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import torch
import os

from yolo.engine.predictor import BasePredictor
from yolo.engine.results import Results
from yolo.utils import ops

from yolo.utils.files import increment_path



class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


class DomainClassifierPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = preds[0]
        #features = preds[1]
        #print(features[0].size(), features[1].size())
        #torch.save(features, os.path.join('/home/eec42/BirdDetector/runs/detect/DAN_pe_palmyra_10percentbkgd_dcgain1.5', 'features.pt'))
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
    

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize)
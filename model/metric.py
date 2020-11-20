from pytorch_lightning.metrics import Metric
import torch
from utils.common_utils import clip_coords

class DetectionMetric(Metric):
    def __init__(self):
        self.height = 100
        self.width = 100
        self.whwh = torch.tensor([width, height, width, height], device=imgs.device)

        # iou threshold 
        iouv = torch.linspace(0.5, 0.95, 10, device=imgs.device)  # iou vector for mAP@0.5:0.95
        # choose the iou threshold for AP computation
        iouv = iouv[0].view(1)  # comment for AP@0.5
        niou = iouv.numel()
        # List to store statistics results
        self.stats = []

    def compute(self):
        return super().compute()

    def update(self):
        return super().update()

    def output_fp_tp_batch(self, pred, target):
        """
        compute true positive and false positive for detection model

        Input:
            pred: predicted boxes in shape (n, 6) (x1, y1, x2, y2, conf, cls)
            target: target boxes in shape (n, 6) (batch_id, cls_id, x, y, w, h)
        """

        if len(self.stats) != 0:
            stats.clear()
        for si, pred in enumerate(pred):
            # select target boxes in current batch_idx
            labels = target[target[:, 0] == si, 1:] # (nl, 5) (cls_id, x, y, w, h)
            # num of labels
            nl = len(labels)
            # target classes (List)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool).numpy(), torch.Tensor().numpy(), torch.Tensor().numpy(), tcls))
                continue

            # Clip boxes to image bounds (inplace operation)
            clip_coords(pred, (self.height, self.width))
            
            # initialize correct tensor
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=pred.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes, resize to image size
                tbox = xywh2xyxy(labels[:, 1:5]) * self.whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    # select the target indices for this specific class; shape (M, )
                    ti = (cls == tcls_tensor).nonzero().view(-1)
                    # select the prediction indices for this class; shape (N, )
                    pi = (cls == pred[:, 5]).nonzero().view(-1) 

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, corresponding target indices
                        # sort the ious, Append detections from the highest iou
                        ious, _ = torch.sort(ious, descending=True)

                        # Append detections (higher than the iou threshold)
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), tcls))

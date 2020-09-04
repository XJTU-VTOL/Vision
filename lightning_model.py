import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.common_utils as utils
from torch.utils.data import DataLoader, DistributedSampler
from utils.common_utils import wh_iou
from models import Darknet
import torch.optim.lr_scheduler as lr_sched
from abc import ABCMeta
import cv2
import numpy as np
import torchvision
import time

class ModelBase(pl.LightningModule):
    def __init__(self, hyper):
        super(ModelBase, self).__init__()
        self.hparams = hyper

    def configure_optimizers(self):
        decay_epochs = [ round(x * self.total_epoch) for x in self.hparams['decay_epoch_list']]

        def lr_lbmd(cur_epoch):
            decay_rate = 1
            for decay_epoch in decay_epochs:
                if cur_epoch > decay_epoch:
                    decay_rate *= self.hparams['lr_decay']
            return max(decay_rate, self.hparams['lrf'] / self.hparams['lr0'])

        updated_params = []
        if hasattr(self, 'conv_layer_index'):
            for i in self.conv_layer_index:
                updated_params += list(self.model.module_list[i].parameters())
        else:
            updated_params = self.parameters()

        self.optimizer = torch.optim.SGD(
            updated_params,
            lr=self.hparams["lr0"],
            weight_decay=self.hparams["weight_decay"],
            momentum=self.hparams['momentum']
        )
        self.lr_scheduler = lr_sched.LambdaLR(self.optimizer, lr_lbmd)

        return  [self.optimizer], [
                {
                 'scheduler': self.lr_scheduler,
                 'interval': 'epoch',
                 'frequency': 1,
                }]

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class YoloLight(ModelBase):

    def __init__(self, opt, hyp, nc, gr=1.0, transfer=False):
        super(YoloLight, self).__init__(hyp)
        self.model = Darknet(opt.cfg, verbose=True)
        self.total_epoch = opt.max_epochs
        self.model.hyp = hyp
        self.model.nc = nc
        self.model.gr = gr

        # transfer learning : only requires grad for the last convolution layer parameters
        if transfer:
            yolo_layer_index = self.model.yolo_layers
            self.conv_layer_index = [i-1 for i in yolo_layer_index]
            for i, module in enumerate(self.model.module_list):
                if i not in self.conv_layer_index:
                    for param in module.parameters():
                        param.requires_grad = False

    def compute_loss(self, p, targets):  # predictions, targets, model
        ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
        lcls, lbox, lobj = 0., 0., 0.
        target_cls, target_box, indices, anchors = self.build_targets(p, targets)  # targets
        h = self.model.hyp  # hyperparameters
        red = 'mean'  # Loss reduction (sum or mean)

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']], device = p[0].device), reduction=red)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']], device = p[0].device), reduction=red)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cls_pos, cls_neg = smooth_BCE(eps=0.0)

        # focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # per output
        num_target = 0  # targets
        for i, pi in enumerate(p):  # layer index, layer predictions
            batch_idx, anchor_idx, gj, gi = indices[i]  # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(pi[..., 0])  # target obj

            num_batch = batch_idx.shape[0]  # number of targets
            if num_batch:
                num_target += num_batch  # cumulative targets

                # select the corresponding shape anchor boxes
                ps = pi[batch_idx, anchor_idx, gj, gi]  # prediction subset corresponding to targets

                # GIoU
                pxy = torch.sigmoid(ps[:, :2])
                pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                giou = bbox_iou(pbox.t(), target_box[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss

                # Obj
                target_obj[batch_idx, anchor_idx, gj, gi] = (1.0 - self.model.gr) + self.model.gr * giou.detach().clamp(0).type(target_obj.dtype)  # giou ratio

                # Class
                if self.model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cls_neg)  # targets
                    t[range(num_batch), target_cls[i]] = cls_pos
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], target_obj)  # obj loss

        lbox *= h['giou']
        lobj *= h['obj']
        lcls *= h['cls']
        if red == 'sum':
            bs = target_obj.shape[0]  # batch size
            g = 3.0  # loss gain
            lobj *= g / bs
            if num_target:
                lcls *= g / num_target / self.model.nc
                lbox *= g / num_target

        loss = lbox + lobj + lcls
        loss_dict = dict(
            lbox = lbox,
            lobj = lobj,
            lcls = lcls,
            loss = loss
        )
        return loss, loss_dict

    def build_targets(self, p, targets):
        """
        Input:
            p: [list] prediction of the network, each for a Yolo Layer
            targets: labels
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_targets = targets.shape[0]
        target_cls, target_box, indices, anchor = [], [], [], []
        # normalized to gridspace scale, because the targets are normalized into 0-1 interval  
        gain = torch.ones(6, device=targets.device)  
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

        style = None
        for i, j in enumerate(self.model.yolo_layers):
            anchors = self.model.module_list[j].anchor_vec
            gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]] - 1  # xyxy gain in current grid scale
            num_anchors = anchors.shape[0]  # number of anchors
            anchor_target = torch.arange(num_anchors).view(num_anchors, 1).repeat(1, num_targets)  # (num_anchors, num_targets) anchor tensor, same as .repeat_interleave(nt)

            # Match targets to anchors
            selected_anchor, offsets = None,  0 # selected anchor index
            targets_gain = targets * gain
            if num_targets:
                # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
                # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
                anchor_target_wh_iou_indice = wh_iou(anchors, targets_gain[:, 4:6]) > self.model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                selected_anchor, targets_gain = anchor_target[anchor_target_wh_iou_indice], targets_gain.repeat(num_anchors, 1, 1)[anchor_target_wh_iou_indice]  # filter

                # overlaps
                gxy = targets_gain[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                if style == 'rect2':
                    g = 0.2  # offset
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

                elif style == 'rect4':
                    g = 0.5  # offset
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                    a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

            # Define
            batch_idx, c = targets_gain[:, :2].long().T  # image, class
            gxy = targets_gain[:, 2:4]  # grid xy
            gwh = targets_gain[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((batch_idx, selected_anchor, gj, gi))  # image, anchor_index, grid indices
            target_box.append(torch.cat((gxy - gij, gwh), 1))  # box (x, y, w, h)
            anchor.append(anchors[selected_anchor])  # anchors
            target_cls.append(c)  # class
            if c.shape[0]:  # if any targets
                assert c.max() < self.model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                            'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                                self.model.nc, self.model.nc - 1, c.max())

        return target_cls, target_box, indices, anchor

    def forward(self, imgs):
        return self.model(imgs)
    
    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        if batch_idx == 0 :
            images = imgs.detach().permute(0, 2, 3, 1).cpu().numpy()
            target = targets.detach().cpu().numpy()
            images = draw_box(images, target)
            self.logger.experiment.add_images('labels epoch %d'%(self.current_epoch), images, dataformats='NHWC')
        pred = self.model(imgs)
        loss, loss_items = self.compute_loss(pred, targets)
        lr = self.optimizer.param_groups[0]['lr']
        loss_items['lr'] = lr
        return {'loss': loss, 'log':loss_items}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([ x['loss'] for x in outputs]).mean()
        return {'loss':avg_loss}

    def validation_epoch_end(self, outputs):
        avg_preision = torch.tensor( [ x['precision'] for x in outputs] ).mean()

        return {'precision': avg_preision}

    def get_boxes(self, prediction, classes=None):
        """
        Performs  Non-Maximum Suppression on inference results
        Returns detections with shape (torch.Tensor):
            N, 6 (x1, y1, x2, y2, conf, cls)
        """
        # Settings
        multi_label = self.hparams['multi_label']
        merge = True  # merge for best mAP
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        
        nc = prediction[0].shape[1] - 5  # number of classes
        multi_label &= nc > 1  # multiple labels per box
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[x[:, 4] > self.hparams['conf_thres']]  # confidence
            x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > self.hparams['conf_thres']).nonzero().t()
                x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1)
                x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

            # Filter by class
            if classes:
                x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c =  x[:, 5]  # classes
            boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.boxes.nms(boxes, scores, self.hparams['iou_thres'])
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > self.hparams['iou_thres']  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                    # i = i[iou.sum(1) > 1]  # require redundancy
                except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                    print(x, i, x.shape, i.shape)
                    pass

            output[xi] = x[i]

        return output
    
    def validation_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        iouv = torch.linspace(0.5, 0.95, 10, device=imgs.device)  # iou vector for mAP@0.5:0.95
        iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        niou = iouv.numel()
        stats = []
        seen = 0

        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.tensor([width, height, width, height], device=imgs.device)

        pred_out, train_out = self.model(imgs)
        output = self.get_boxes(pred_out)

        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool).numpy(), torch.Tensor().numpy(), torch.Tensor().numpy(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=pred.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            if niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.model.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
        
        metric = dict(
            precision = mp,
            recall = mr,
            ap = map,
            f1 = mf1
        )

        return {'precision': mp, 'log': metric}
    
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def draw_box(imgs, target):
    """
        imgs : numpy array B * H * W * 3
        target : numpy array N * 6
    """
    h, w = imgs.shape[1:3]
    boxes = []
    for index, img in enumerate(imgs):
        batch_target_index = np.where(target[:, 0] == index)
        batch_target = target[batch_target_index] # N * 6
        for box in batch_target:
            center = (int(box[2]*w), int(box[3]*h))
            width = int(box[4] * w)
            height = int(box[5] * h)
            img = cv2.rectangle(img, (center[0] - width // 2, center[1] - height // 2), 
                                (center[0] + width // 2, center[1] + height //2), color=(255, 0, 0), thickness=4)
        boxes.append(img)
    return np.stack(boxes, axis=0)

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x, device=x.device) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

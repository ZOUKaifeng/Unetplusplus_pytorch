
import os
import gc
import cv2
import time
import math
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm 
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tq
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import *
import segmentation_models_pytorch as smp

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils import data
from torch.optim.optimizer import Optimizer, required
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from time import time
import albumentations 



####################### data augmentations ################################
AUGMENTATIONS_TRAIN =  albumentations.Compose([
        albumentations.Resize(350, 525),
        albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        ]),
        albumentations.OneOf([
            albumentations.Blur(blur_limit=4, p=1),
            albumentations.MotionBlur(blur_limit=4, p=1),
            albumentations.MedianBlur(blur_limit=4, p=1)
        ], p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
],p=1)

AUGMENTATIONS_TEST =  albumentations.Compose([
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
],p=1)



#########################data loader#####################################

def rle_to_mask(rle_string, height = 1400 , width = 2100):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 1
        img = img.reshape(cols,rows)
        img = img.T
        return img

def make_mask(df: pd.DataFrame, image_name: str = "img.jpg"):  
    """
    Create mask based on df, image name and shape.
    """
    df = df[df['ImageId'] == image_name] 
    test = []
    masks = np.zeros((1400, 2100, 4))
    for idx, encode_pixels in enumerate(df['Label_EncodedPixels']):
        mask = rle_to_mask(encode_pixels[1])
        masks[:,:,idx] = mask
    return masks



def default_loader(id, mode, df):

    img = np.array(Image.open(params['ROOT']+'/train/' + id).convert('RGB'))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    
    
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    #name = id.split('/')[-1]
    #name = name.split('.')
    mask = make_mask(df, id)
    #mask_path = os.path.join(os.path.abspath(os.path.join(id, "../..")), 'mask', name[0] + '.png')
    #mask = np.asarray(Image.open(mask_path), np.uint8)
    #mask = np.asarray(mask, np.int32)
    #image_name = id.split('/')[-1]
    #label = make_mask(df, image_name)

    if mode == 'train':
        augmented = AUGMENTATIONS_TRAIN(image = img, mask = mask)
        img = augmented['image']
        mask = augmented['mask']

    else:
        augmented = AUGMENTATIONS_TEST(image = img, mask = mask)
        img = augmented['image']
        mask = augmented['mask']

    #mask = np.expand_dims(mask, axis=0)
    img = cv2.resize(img, (640, 320), cv2.INTER_NEAREST)
    mask = cv2.resize(mask,(640,320), cv2.INTER_NEAREST)
    img = img.transpose(2,0,1)
    mask = mask.transpose(2,0,1)
    return img, mask

class CloudDataset(data.Dataset):

    #def __init__(self, trainlist, root):
    def __init__(self, imagelist, mode, df):
        self.imagelist = imagelist
        self.loader = default_loader

        self.mode = mode
        self.df = df
    def __getitem__(self, index):

#        id = self.imagelist[index]
        img, mask = self.loader(self.imagelist[index], self.mode, self.df)
        return img, mask

    def __len__(self):
        return len(self.imagelist)


########################Metric#####################################
def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())

def dice_with_threshold(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
):
    """
    Reference:
    https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/utils/criterion/dice.html
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice

class Metric:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, epoch, phase):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold

        self.dice = []
        self.iou_scores = []

    def update(self, outputs, targets, threshold = 0.5):

        probs = torch.sigmoid(outputs)
        preds = (probs > self.base_threshold).float()

        dice_score = dice_with_threshold(probs, targets, threshold = threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])

        self.iou_scores.append(iou)
        self.dice.append(dice_score)

    def get_metrics(self):
        dice = np.mean(self.dice)

        iou = np.mean(self.iou_scores, axis = 0)
        return dice, iou

def epoch_log(phase, epoch, epoch_loss, meter, mylog):
    '''logging the metrics at the end of an epoch'''
    dice, iou = meter.get_metrics()
    
    print(phase + "  Loss: %0.4f | IoU: %0.4f | dice: %0.4f | iou_0: %0.4f | iou_1: %0.4f | iou_2: %0.4f | iou_3: %0.4f" 
                       % (epoch_loss, iou.mean(), dice, iou[0], iou[1], iou[2], iou[3]))
    print(phase + "  Loss: %0.4f | IoU: %0.4f | dice: %0.4f | iou_0: %0.4f | iou_1: %0.4f | iou_2: %0.4f | iou_3: %0.4f" 
                       % (epoch_loss, iou.mean(), dice, iou[0], iou[1], iou[2], iou[3]), file = mylog) 
    return dice, iou

def compute_ious(preds, labels, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    #pred[label == ignore_index] = 0
    ious = []
    for pred, label in zip(preds, labels):
        if np.sum(pred) == 0 and np.sum(label) == 0:
            ious.append(1)
            continue
        intersection = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()
        if union != 0:
            ious.append(intersection / union)
        else:
            ious.append(0)
    return ious 

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np

    for pred, label in zip(preds, labels):
        ious.append(compute_ious(pred, label, classes))
    iou = np.mean(ious, axis = 0)
    return iou

####################################Loss##############################################
def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)


#####################optimizer######################
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

###############################train##############################
def train(model, train_data_loader, criterion, optimizer, epoch, mylog):
    model.train()
    data_iter = iter(train_data_loader)
    train_epoch_loss = 0
    count = 0
    train_metric = Metric(epoch, 'train')
    for img, mask in data_iter:
        count += 1
        img = img.cuda()
        mask = mask.cuda()
        optimizer.zero_grad()
        ###########forward##################
        logit = model(img)
        loss = criterion(logit, mask.float())
        ###########backpropagation###############
        loss.backward()
        optimizer.step()
        #########metric###################
        train_epoch_loss += loss
        with torch.no_grad():
            train_metric.update(logit.cpu(), mask.cpu().float())
            print('epoch:[%d][%d/%d]|loss: %0.4f (%0.4f) |' % ( epoch, count, len(data_iter), loss.cpu().detach().numpy(), train_epoch_loss/count))
    train_epoch_loss /= count
    dice, IoU= epoch_log('train', epoch, train_epoch_loss, train_metric, mylog)
    return train_epoch_loss, dice, IoU


def valid(model, val_data_loader, criterion, optimizer, epoch, mylog):
    data_iter = iter(val_data_loader)
    model.eval()
    val_metric = Metric(epoch, 'val')
    val_epoch_loss = 0
    for img, mask in tqdm(data_iter):
        img = img.cuda()
        mask = mask.cuda()
        with torch.no_grad():
            logit = model(img)
            loss = criterion(logit, mask.float())
            val_metric.update(logit.cpu(), mask.cpu().float())
        val_epoch_loss += loss

    valid_epoch_loss = val_epoch_loss / len(data_iter)

    dice, IoU = epoch_log('valid', epoch, valid_epoch_loss, val_metric, mylog)

    return valid_epoch_loss, dice, IoU

    


def main():
    if not os.path.exists(params['work_dir']):
        os.makedirs(params['work_dir'])
    if not os.path.exists(params['weight_path']):
            os.makedirs(params['weight_path'])
    if not os.path.exists(params['output']):
            os.makedirs(params['output'])

    
    train_df = pd.read_csv(params['ROOT'] + 'train.csv').fillna(-1)
    #train_df = pd.read_csv(train_csv_path).fillna(-1)
    # image id and class id are two seperate entities and it makes it easier to split them up in two columns
    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
    #train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
    # lets create a dict with class id and encoded pixels and group all the defaults per image
    train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)

    imagelist = os.listdir(os.path.join(params['ROOT'],'train'))

    val_ratio = 0.12
    train_list, val_list = train_test_split(imagelist, test_size=val_ratio, random_state=666)

    train_data = CloudDataset(train_list, 'train', train_df)
    val_data = CloudDataset(val_list, 'val', train_df)
    ############################dataloader####################################
    print('data size:', len(imagelist))
    print('train data:', len(train_data))
    print('valid data:',len(val_data))


    train_data_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size = params['batch_size'],
                    shuffle=True,
                    num_workers=params['num_workers'])
    
    val_data_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size = params['batch_size'],
                    shuffle = True,
                    num_workers = params['num_workers'],
                    )

    #######################initialize model##############################
    ENCODER = 'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=4, 
        activation=None,
        )

    ###################log file#########################
    mylog = open(params['log'],'a')
    tic = time()
    ################some information
    print('data:', params['ROOT'])
    print('weight:',params['weight_path'])

    print('data:', params['ROOT'], file = mylog)
    print('weight:',params['weight_path'], file = mylog)


    ######################optimizer config########################

    criterion = BCEDiceLoss(eps=1.0, activation='sigmoid')
    optimizer = RAdam(model.parameters(), lr = params['learning_rate'])
    #current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    best_loss = 0.8250
    best_acc = 0.5769
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=params['gpus'])
    if params['load_checkpoint']:
        model.load_state_dict(torch.load(params['checkpoint']))
        print('load checkpoint from', params['checkpoint'])
 
    #####################start train##############################
    for epoch in range(1,  params['epoch_num'] + 1):
        print ("epoch:{0}   time:{1}".format(epoch, int(time()-tic)))
        print("learning rate: %g" % optimizer.param_groups[0]['lr'])
        print ("epoch:{0}   time:{1}".format(epoch, int(time()-tic)), file = mylog)
        print("learning rate: %g" % optimizer.param_groups[0]['lr'], file = mylog)

    ##################train model##################################
        train_loss, train_dice, train_IoU = train(model, train_data_loader, criterion, optimizer, epoch, mylog)

    ##################valid#########################################
        valid_loss, valid_dice, valid_IoU = valid(model, val_data_loader, criterion, optimizer, epoch, mylog)

        scheduler.step(valid_loss)

        if valid_loss <= best_loss:
            best_loss = valid_loss
            print('save to =>>', params['lowest_loss'])
            print('save to =>>', params['lowest_loss'], file = mylog)
            torch.save(model.state_dict(), params['lowest_loss'])
        if valid_dice >= best_acc:
            best_acc = valid_dice
            print('save to =>>', params['best_acc'])
            print('save to =>>', params['best_acc'], file = mylog)
            torch.save(model.state_dict(), params['best_acc'])
        torch.save(model.state_dict(), params['checkpoint'])
        mylog.flush()


if __name__ == '__main__':
    main()


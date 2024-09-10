
import json
import os
import gc
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from src.utils import setup_seed
from src.pixel_classifier import pixel_classifier
from src.feature_extractors import create_feature_extractor, collect_features
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion import dev

from utils import read_text
from src.load_dataset import RandomGenerator, ValGenerator, Mixdataset
import shutil



def logger_config(log_path):
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, preds, target, weight=None, softmax=False):
        if softmax:
            preds = torch.softmax(preds, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert preds.size() == target.size(), 'predict {} & target {} shape do not match'.format(preds.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(preds[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        return loss / self.n_classes


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef_bce =  1.0
        self.coef_dice = 1.5

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        diceloss = 1 - dice.sum() / num
        return self.coef_bce * bce + self.coef_dice * diceloss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    return iou, dice


def evaluation(args, model, extractor, valloader ):
    device =dev()
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=device)
    else:
        noise = None
    preds, gts, uncertainty_scores = [], [], []
    for idx, sample in enumerate(valloader):
        img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
        img = img.to(device)
        text = text.to(device)
        features = extractor(img, noise=noise)
        features = collect_features(activations=features)

        for k, v in features.items():
            features[k] = features[k].to(text.device)
        with torch.no_grad():
            pred = model(features, text)
            assert pred.dim() == 4 and pred.shape[1] > 1, "pred outputs should have >1 classes"
            pred_softmax = torch.softmax(pred, dim=1)
            _, pred = torch.max(pred_softmax, dim=1)

        gts.append(label.numpy())
        preds.append(pred.cpu().numpy())

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    for pred, target in zip(preds, gts):
        iou, dice = iou_score(pred, target)
        iou_avg_meter.update(iou, target.shape[0])
        dice_avg_meter.update(dice, target.shape[0])

    return dice_avg_meter.avg, iou_avg_meter.avg


def main(args, extractor, data_loader):
    device = dev()
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=device)
    else:
        noise = None

    gc.collect()
    extract_dims = [v * len(opts['steps']) for v in opts['dim']]
    classifier = pixel_classifier(extract_dims=extract_dims)

    classifier.init_weights()
    classifier = classifier.cuda()
    criterion_cross_entro = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=2)
    # criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60], gamma=0.1)

    stats = {'best_dice': 0., 'best_iou': 0., 'best_epoch': 0, 'best_ckpt': None}

    for epoch in range(args['max_training']):
        classifier.train()
        for idx, sample in enumerate(data_loader):
            img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
            img = img.to(device)
            label = label.to(device)
            text = text.to(device)
            features = extractor(img, noise=noise)
            features = collect_features(activations=features)
            for k, v in features.items():
                features[k] = features[k].to(device)
            y_pred = classifier(features, textf=text)
            y_batch = label.type(torch.long)
            optimizer.zero_grad()
            loss = criterion_cross_entro(y_pred, y_batch)
            loss += 1.5 * criterion_dice(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            eval_dice, eval_iou = evaluation(opts, classifier.eval(), extractor= extractor, valloader=val_loader)
            if eval_dice > stats['best_dice']:
                stats['best_dice'] = eval_dice
                stats['best_iou'] = eval_iou
                stats['best_epoch'] = epoch
                stats['best_ckpt'] = classifier.state_dict()
                model_path = os.path.join(args['exp_dir'], 'model_' + f'{epoch:02d}.pth')
                torch.save({'model_state_dict': stats['best_ckpt']}, model_path)

            logger.info(f"Epoch {epoch:02d}: dice/iou= {eval_dice:.4f}/{eval_iou:.4f} ")

    saved_path = os.path.join(args['exp_dir'], 'model_best.pth')
    logger.info(f'final model saved to: {saved_path} \n best_epoch:{stats["best_epoch"]} best_dice:{stats["best_dice"]:.4f} best_iou:{stats["best_iou"]:.4f}')
    shutil.copy(src=model_path, dst=saved_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=40)

    args = parser.parse_args()
    setup_seed(args.seed)
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))

    os.makedirs(opts['exp_dir'], exist_ok=True)
    opts['exp_dir'] = os.path.join(opts['exp_dir'], f'experiment-{len(os.listdir(opts["exp_dir"]))+ 1:02d}')
    os.makedirs(opts['exp_dir'], exist_ok=True)
    print('Experiment folder: %s' % (opts['exp_dir']))
    shutil.copy(args.exp, opts['exp_dir'])

    train_text = read_text(os.path.join(opts['training_path'], 'Train_text.xlsx'))
    train_tf = RandomGenerator(output_size=[opts['image_size'], opts['image_size']])
    train_dt = Mixdataset(dataset_path=opts['training_path'], row_text=train_text, joint_transform=train_tf,)
    val_text = read_text(os.path.join(opts['validation_path'], 'Val_text.xlsx'))
    val_tf = ValGenerator(output_size=[opts['image_size'], opts['image_size']])
    val_dt = Mixdataset(dataset_path=opts['validation_path'], row_text=val_text, joint_transform=val_tf,)

    logger = logger_config(os.path.join(opts['exp_dir'], 'train.log'))
    train_loader = DataLoader(dataset=train_dt, batch_size=opts['batch_size'], shuffle=True,  drop_last=True)
    val_loader = DataLoader(dataset=val_dt, batch_size=opts['batch_size'], shuffle=False,  drop_last=True)

    fea_extractor = create_feature_extractor(**opts)

    main(opts, extractor=fea_extractor, data_loader=train_loader)


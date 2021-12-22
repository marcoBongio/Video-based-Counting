import math
import os

from matplotlib import pyplot as plt, cm
from torchinfo import summary

from model import CANNet2s
from utils import save_checkpoint

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
import argparse
import json
import cv2
import dataset
import time

from variables import HEIGHT, WIDTH, MODEL_NAME, PATCH_SIZE_PF, MEAN, STD

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')


def plotDensity(density, axarr, k):
    '''
    @density: np array of corresponding density map
    '''
    density = density * 255.0

    # plot with overlay
    colormap_i = cm.jet(density)[:, :, 0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]

    axarr[k].imshow(255 * new_map.astype(np.uint8))


def main():
    global args

    args = parser.parse_args()
    args.best_prec1 = 1e6
    args.lr = 1e-4
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.start_frame = 0
    args.epochs = 200
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 1
    args.log_freg = 3600

    with open(args.train_json, 'r') as outfile:
        args.train_list = json.load(outfile)
    with open(args.val_json, 'r') as outfile:
        args.val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    model = CANNet2s()

    model = model.cuda()

    criterion = nn.MSELoss(reduction='sum').cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.decay)

    summary(model, input_size=((args.batch_size, 3, HEIGHT, WIDTH), (args.batch_size, 3, HEIGHT, WIDTH)))

    # modify the path of saved checkpoint if necessary
    try:
        checkpoint = torch.load('models/checkpoint_' + MODEL_NAME + '.pth.tar', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer)
        args.start_epoch = checkpoint['epoch']
        args.start_frame = checkpoint['start_frame']
        try:
            args.best_prec1 = checkpoint['best_prec'].item()
        except:
            args.best_prec1 = checkpoint['best_prec']
        print("Train model " + MODEL_NAME + " from epoch " + str(args.start_epoch) + " with best prec = " + str(
            args.best_prec1) + "...")
    except:
        print("Train model " + MODEL_NAME + "...")

    for epoch in range(args.start_epoch, args.epochs):
        train(args.train_list, model, criterion, optimizer, epoch)
        prec1 = validate(args.val_list, model, criterion)

        is_best = prec1 < args.best_prec1
        args.best_prec1 = min(prec1, args.best_prec1)
        args.start_frame = 0
        print(' * best MSE {mse:.3f} '
              .format(mse=args.best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'start_frame': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec': args.best_prec1
        }, is_best)


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                                            std=STD),
                            ]),
                            train=True,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset) + args.start_frame, args.lr))

    model.train()
    end = time.time()

    for i, (prev_img, img, post_img, prev_target, target, post_target) in enumerate(train_loader):
        if i + 1 <= args.start_frame:
            continue
        data_time.update(time.time() - end)

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        post_img = post_img.cuda()
        post_img = Variable(post_img)

        prev_flow, _ = model(prev_img, img)
        post_flow, _ = model(img, post_img)

        prev_flow_inverse, _ = model(img, prev_img)
        post_flow_inverse, _ = model(post_img, img)

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        prev_target = prev_target.type(torch.FloatTensor)[0].cuda()
        prev_target = Variable(prev_target)

        post_target = post_target.type(torch.FloatTensor)[0].cuda()
        post_target = Variable(post_target)

        # mask the boundary locations where people can move in/out between regions outside image plane
        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :],
                                                                                        (0, 0, 0, 1)) + F.pad(
            prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :,
                                                                                                    :] + F.pad(
            prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
            prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :,
                                                                                                      :] * mask_boundry

        reconstruction_from_post = torch.sum(post_flow[0, :9, :, :], dim=0) + post_flow[0, 9, :, :] * mask_boundry

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :,
                                                                                              :] * mask_boundry

        reconstruction_from_post_inverse = F.pad(post_flow_inverse[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(
            post_flow_inverse[0, 1, 1:, :], (0, 0, 0, 1)) + F.pad(post_flow_inverse[0, 2, 1:, :-1],
                                                                  (1, 0, 0, 1)) + F.pad(post_flow_inverse[0, 3, :, 1:],
                                                                                        (0, 1, 0,
                                                                                         0)) + post_flow_inverse[0, 4,
                                                                                               :, :] + F.pad(
            post_flow_inverse[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow_inverse[0, 6, :-1, 1:],
                                                                   (0, 1, 1, 0)) + F.pad(
            post_flow_inverse[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow_inverse[0, 8, :-1, :-1],
                                                                   (1, 0, 1, 0)) + post_flow_inverse[0, 9, :,
                                                                                   :] * mask_boundry

        prev_density_reconstruction = torch.sum(prev_flow[0, :9, :, :], dim=0) + prev_flow[0, 9, :, :] * mask_boundry
        prev_density_reconstruction_inverse = F.pad(prev_flow_inverse[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(
            prev_flow_inverse[0, 1, 1:, :], (0, 0, 0, 1)) + F.pad(prev_flow_inverse[0, 2, 1:, :-1],
                                                                  (1, 0, 0, 1)) + F.pad(prev_flow_inverse[0, 3, :, 1:],
                                                                                        (0, 1, 0,
                                                                                         0)) + prev_flow_inverse[0, 4,
                                                                                               :, :] + F.pad(
            prev_flow_inverse[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow_inverse[0, 6, :-1, 1:],
                                                                   (0, 1, 1, 0)) + F.pad(
            prev_flow_inverse[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow_inverse[0, 8, :-1, :-1],
                                                                   (1, 0, 1, 0)) + prev_flow_inverse[0, 9, :,
                                                                                   :] * mask_boundry

        post_density_reconstruction_inverse = torch.sum(post_flow_inverse[0, :9, :, :], dim=0) + post_flow_inverse[0, 9,
                                                                                                 :, :] * mask_boundry
        post_density_reconstruction = F.pad(post_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(post_flow[0, 1, 1:, :],
                                                                                           (0, 0, 0, 1)) + F.pad(
            post_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(post_flow[0, 3, :, 1:], (0, 1, 0, 0)) + post_flow[0, 4, :,
                                                                                                    :] + F.pad(
            post_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
            post_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + post_flow[0, 9, :,
                                                                                                      :] * mask_boundry

        prev_reconstruction_from_prev = torch.sum(prev_flow[0, :9, :, :], dim=0) + prev_flow[0, 9, :, :] * mask_boundry
        post_reconstruction_from_post = F.pad(post_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(post_flow[0, 1, 1:, :],
                                                                                             (0, 0, 0, 1)) + F.pad(
            post_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(post_flow[0, 3, :, 1:], (0, 1, 0, 0)) + post_flow[0, 4, :,
                                                                                                    :] + F.pad(
            post_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
            post_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + post_flow[0, 9, :,
                                                                                                      :] * mask_boundry

        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)
        loss_prev = criterion(prev_reconstruction_from_prev, prev_target)
        loss_post = criterion(post_reconstruction_from_post, post_target)

        # cycle consistency
        loss_prev_consistency = criterion(prev_flow[0, 0, 1:, 1:], prev_flow_inverse[0, 8, :-1, :-1]) + criterion(
            prev_flow[0, 1, 1:, :], prev_flow_inverse[0, 7, :-1, :]) + criterion(prev_flow[0, 2, 1:, :-1],
                                                                                 prev_flow_inverse[0, 6, :-1,
                                                                                 1:]) + criterion(
            prev_flow[0, 3, :, 1:], prev_flow_inverse[0, 5, :, :-1]) + criterion(prev_flow[0, 4, :, :],
                                                                                 prev_flow_inverse[0, 4, :,
                                                                                 :]) + criterion(
            prev_flow[0, 5, :, :-1], prev_flow_inverse[0, 3, :, 1:]) + criterion(prev_flow[0, 6, :-1, 1:],
                                                                                 prev_flow_inverse[0, 2, 1:,
                                                                                 :-1]) + criterion(
            prev_flow[0, 7, :-1, :], prev_flow_inverse[0, 1, 1:, :]) + criterion(prev_flow[0, 8, :-1, :-1],
                                                                                 prev_flow_inverse[0, 0, 1:, 1:])

        loss_post_consistency = criterion(post_flow[0, 0, 1:, 1:], post_flow_inverse[0, 8, :-1, :-1]) + criterion(
            post_flow[0, 1, 1:, :], post_flow_inverse[0, 7, :-1, :]) + criterion(post_flow[0, 2, 1:, :-1],
                                                                                 post_flow_inverse[0, 6, :-1,
                                                                                 1:]) + criterion(
            post_flow[0, 3, :, 1:], post_flow_inverse[0, 5, :, :-1]) + criterion(post_flow[0, 4, :, :],
                                                                                 post_flow_inverse[0, 4, :,
                                                                                 :]) + criterion(
            post_flow[0, 5, :, :-1], post_flow_inverse[0, 3, :, 1:]) + criterion(post_flow[0, 6, :-1, 1:],
                                                                                 post_flow_inverse[0, 2, 1:,
                                                                                 :-1]) + criterion(
            post_flow[0, 7, :-1, :], post_flow_inverse[0, 1, 1:, :]) + criterion(post_flow[0, 8, :-1, :-1],
                                                                                 post_flow_inverse[0, 0, 1:, 1:])

        loss = loss_prev_flow + loss_post_flow + loss_prev_flow_inverse + loss_post_flow_inverse + loss_prev + loss_post + loss_prev_consistency + loss_post_consistency

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            print("\nTarget = " + str(torch.sum(target)))
            overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0).data.cpu().numpy()
            pred_sum = overall.sum()
            print("Pred = " + str(pred_sum))
            print("Reconstruction from prev = " + str(torch.sum(reconstruction_from_prev)))
            print("Reconstruction from post = " + str(torch.sum(reconstruction_from_post)))
            print("Reconstruction from prev inverse = " + str(torch.sum(reconstruction_from_prev_inverse)))
            print("Reconstruction from post inverse = " + str(torch.sum(reconstruction_from_post_inverse)))
            print("Prev Target = " + str(torch.sum(prev_target)))
            print("Prev Reconstruction from prev = " + str(torch.sum(reconstruction_from_prev)))
            print("Post Target = " + str(torch.sum(post_target)))
            print("Post Reconstruction from post = " + str(torch.sum(reconstruction_from_prev)) + "\n")

            print("loss_prev_flow = " + str(loss_prev_flow))
            print("loss_post_flow = " + str(loss_post_flow))
            print("loss_prev_flow_inverse = " + str(loss_prev_flow_inverse))
            print("loss_post_flow_inverse = " + str(loss_post_flow_inverse))
            print("loss_prev = " + str(loss_prev))
            print("loss_post = " + str(loss_post))
            print("loss_prev_consistency = " + str(loss_prev_consistency))
            print("loss_post_consistency = " + str(loss_post_consistency))

            pred = cv2.resize(overall, (overall.shape[1] * PATCH_SIZE_PF, overall.shape[0] * PATCH_SIZE_PF),
                              interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)

            target = cv2.resize(target.cpu().detach().numpy(),
                                (target.shape[1] * PATCH_SIZE_PF, target.shape[0] * PATCH_SIZE_PF),
                                interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
            fig, axarr = plt.subplots(1, 2)
            plotDensity(pred, axarr, 0)
            plotDensity(target, axarr, 1)
            plt.show()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        if ((i + 1) % args.log_freg == 0) & ((i + 1) != len(train_loader)):
            prec1 = validate(args.val_list, model, criterion)

            is_best = prec1 < args.best_prec1
            args.best_prec1 = min(prec1, args.best_prec1)
            print(' * best MSE {mse:.3f} '
                  .format(mse=args.best_prec1))
            save_checkpoint({
                'epoch': epoch,
                'start_frame': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec': args.best_prec1
            }, is_best)


def validate(val_list, model, criterion):
    print('begin val')
    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                                            std=STD),
                            ]),
                            train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0.0
    mse = 0.0

    for i, (prev_img, img, post_img, _, target, _) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        with torch.no_grad():
            prev_flow, _ = model(prev_img, img)
            prev_flow_inverse, _ = model(img, prev_img)

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :],
                                                                                        (0, 0, 0, 1)) + F.pad(
            prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :,
                                                                                                    :] + F.pad(
            prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
            prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :,
                                                                                                      :] * mask_boundry

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :,
                                                                                              :] * mask_boundry

        overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0).type(torch.FloatTensor)

        target = target.type(torch.FloatTensor)

        if i % args.print_freq == 0:
            print("PRED = " + str(overall.data.sum()))
            print("GT = " + str(target.sum()))
        mae += abs(overall.data.sum() - target.sum())
        mse += abs(overall.data.sum() - target.sum()) * abs(overall.data.sum() - target.sum())

    mae = mae / len(val_loader)
    mse = math.sqrt(mse / len(val_loader))
    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    print(' * MSE {mse:.3f} '
          .format(mse=mse))

    return mse


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


if __name__ == '__main__':
    main()

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import cv2

from model import *
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=False, type=bool, help='Use CUDA to train model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')

args = parser.parse_args()

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

torch.set_default_tensor_type('torch.FloatTensor')  # Ensure all tensors default to CPU

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def save_video_frames(tensor, path):
    frames = tensor.cpu().float().numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))  # Change to (num_frames, height, width, channels)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 20.0, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        frame = (frame * 255.0).astype('uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Network(stage=args.stage)

    model.enhance.in_conv.apply(model.weights_init)
    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    model.calibrate.in_conv.apply(model.weights_init)
    model.calibrate.convs.apply(model.weights_init)
    model.calibrate.out_conv.apply(model.weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)

    train_video_dir = 'D:/The SquadSync/IEM-main_2/data/Videos'
    TrainDataset = MemoryFriendlyLoader(video_dir=train_video_dir, task='train')

    test_video_dir = './data/medium'
    TestDataset = MemoryFriendlyLoader(video_dir=test_video_dir, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True)

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True)

    total_step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False)

            optimizer.zero_grad()
            loss = model._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_losses.append(loss.item())
            logging.info('train-epoch %03d %03d %f', epoch, batch_idx, loss)

        # Calculate average training loss for the epoch
        avg_train_loss = np.average(train_losses)
        logging.info('train-epoch %03d %f', epoch, avg_train_loss)

        # Validate on the test set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _, (input, _) in enumerate(test_queue):
                input = Variable(input, requires_grad=False)
                val_loss = model._loss(input)
                val_losses.append(val_loss.item())

        avg_val_loss = np.average(val_losses)
        logging.info('val-epoch %03d %f', epoch, avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            utils.save(model, os.path.join(model_path, 'best_weights.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info('Early stopping at epoch %03d', epoch)
                break

        # Save model at current epoch
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        # Save video frames for each epoch
        if epoch % 1 == 0 and total_step != 0:
            logging.info('train %03d %f', epoch, avg_train_loss)
            with torch.no_grad():
                for _, (input, video_name) in enumerate(test_queue):
                    input = Variable(input, requires_grad=False)
                    video_name = video_name[0].split('\\')[-1].split('.')[0]
                    illu_list, ref_list, input_list, atten = model(input)
                    u_name = '%s.mp4' % (video_name + '_' + str(epoch))
                    u_path = image_path + '/' + u_name
                    save_video_frames(ref_list[0], u_path)

if __name__ == '__main__':
    main()

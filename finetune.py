import os
import sys
import numpy as np
from PIL import Image
import logging
import argparse
import torch.utils
from torch.autograd import Variable
import cv2

from model import *
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--steps', type=float, default=100, help='finetune steps')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--save', type=str, default='results/finetune/', help='location to save output')
parser.add_argument('--model', type=str, default='./weights/difficult.pt', help='location of the pre-trained model')

args = parser.parse_args()

# Set directories for training and testing videos directly in the code
train_dir = 'D:/The SquadSync/IEM-main_2/data/Videos'  # Path to the directory with training videos
test_dir = 'D:/The SquadSync/IEM-main_2/data/test_video'    # Path to the directory with testing videos

os.makedirs(args.save, exist_ok=True)
torch.manual_seed(args.seed)

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def save_video_frames(tensor, path):
    frames = tensor.cpu().float().numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 20.0, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        frame = (frame * 255.0).astype('uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

def main():
    logging.info("args = %s", args)

    # Load the model with weights mapped to CPU
    base_weights = torch.load(args.model, map_location=torch.device('cpu'),weights_only=True)
    model = Finetunemodel(base_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    
    # Use the specified training and testing directories
    TrainDataset = MemoryFriendlyLoader(video_dir=train_dir, task='train')
    TestDataset = MemoryFriendlyLoader(video_dir=test_dir, task='test')

    # Check if datasets are empty
    if len(TrainDataset) == 0:
        logging.error("Training dataset is empty. Please check the directory: %s", train_dir)
        return
    if len(TestDataset) == 0:
        logging.error("Testing dataset is empty. Please check the directory: %s", test_dir)
        return

    train_queue = torch.utils.data.DataLoader(TrainDataset, batch_size=args.batch_size, shuffle=True)
    test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, shuffle=True)

    total_step = 0

    model.train()
    for step in range(args.steps):
        for batch_idx, (input, _) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False)

            optimizer.zero_grad()
            loss = model._loss(input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            print('finetune-step:{} loss:{}'.format(step, loss.item()))

            if total_step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    for _, (input, image_name) in enumerate(test_queue):
                        input = Variable(input)
                        image_name = image_name[0].split('\\')[-1].split('.')[0]
                        _, ref = model(input)

                        u_name = '%s.png' % (image_name + '_' + str(total_step) + '_ref_')
                        u_path = os.path.join(args.save, u_name)
                        save_images(ref, u_path)

if __name__ == '__main__':
    main()

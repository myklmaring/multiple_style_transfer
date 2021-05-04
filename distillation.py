import numpy as np
import time
import argparse
import os
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformer import Transformer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.datasets as datasets
from vgg import VGG16


def train(args):
    print("Start Time:\t{}".format(time.ctime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = Transformer()
    model2 = Transformer()
    state_dict1 = torch.load(args.model1)
    state_dict2 = torch.load(args.model2)
    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)
    model1.to(device)
    model2.to(device)
    vgg = VGG16().to(device)

    train_dataset = datasets.ImageFolder(args.datapath, transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    transformer = Transformer(norm='instance', padding='reflect').to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    loss = []
    run_time = time.strftime("%d-%H-%M-%S")

    for epoch_num in range(args.epochs):
        transformer.train()
        agg_one_loss = 0.0
        agg_two_loss = 0.0
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            content = x.to(device)

            y_hat = transformer(content)
            y_model1 = model1(content)
            y_model2 = model2(content)

            features_yh = vgg(normalize(y_hat))
            features_y1 = vgg(normalize(y_model1))
            features_y2 = vgg(normalize(y_model2))


            # Do this but with losses from the output of the VGG blocks
            # one_loss = mse_loss(y_hat, y_model1)
            # two_loss = mse_loss(y_hat, y_model2)
            one_loss = sum(np.array([mse_loss(feat_yh, feat_y1) for feat_yh, feat_y1 in zip(features_yh.values(), features_y1.values())]))
            two_loss = sum(np.array([mse_loss(feat_yh, feat_y2) for feat_yh, feat_y2 in zip(features_yh.values(), features_y2.values())]))

            total_loss = one_loss + two_loss
            total_loss.backward()
            optimizer.step()

            agg_one_loss += one_loss.item()
            agg_two_loss += two_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "[{}/{}]\tTotal: {:.2f}\tModel 1: {:.2f}\tModel 2: {:.2f}".format(
                    count,
                    len(train_dataset),
                    (agg_one_loss + agg_two_loss) / (batch_id + 1),
                    agg_one_loss / (batch_id + 1),
                    agg_two_loss/(batch_id+1),
                )
                print(mesg)

                loss.append([batch_id+1,
                             agg_one_loss / (batch_id + 1),
                             agg_two_loss / (batch_id + 1),
                             (agg_one_loss + agg_two_loss) / (batch_id + 1)])

            if args.checkpoint_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch_num+1) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()
                save_loss_plot(np.array(loss), args.log_dir + '/train_loss{}.jpg'.format(run_time))

    # save model and parameter log
    transformer.eval().cpu()

    if args.savename is None:
        save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.strftime("%d-%H-%M-%S")) + ".model"
    else:
        save_model_filename = args.savename

    save_model_path = os.path.join(args.save_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    # save loss in pickle file
    with open('{}/loss{}'.format(args.log_dir, run_time), 'wb') as fp:
        pickle.dump(loss, fp)

    with open('{}/param_log{}.txt'.format(args.log_dir, run_time), 'w') as f:
        f.write("Epochs: {}\n".format(args.epochs))
        f.write("Batch Size: {}\n".format(args.batch_size))
        f.write("Dataset: {}\n".format(args.datapath))
        f.write("Learning Rate: {}\n".format(args.lr))
        f.write("Model 1: {}\n".format(args.model1))
        f.write("Model 2: {}\n".format(args.model2))

    print("\nDone, trained model saved at", save_model_path)


def save_loss_plot(loss, path):
    iters = loss[:, 0]
    total = loss[:, 3]
    content = loss[:, 1]
    style = loss[:, 2]

    plt.plot(iters, total, 'k-', label='Total Loss', linewidth=2)
    plt.plot(iters, content, 'r-', label='Model 1 Loss', linewidth=2)
    plt.plot(iters, style, 'b-', label='Model 2 Loss', linewidth=2)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.savefig(path)
    plt.close()


def check_urself(args):
    # check for checkpoint dir, save dir, log dir/ otherwise make them
    try:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
    except OSError:
        print("Cry because nothing worked")

    # data path, style path
    if not os.path.exists(args.datapath):
        print("You better change something. I couldn't find your training dataset")

    return


def normalize(tens):
    # normalize images according to ImageNet mean and std
    mean = tens.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = tens.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (tens.div_(255.0)-mean)/std


def main():
    parser = argparse.ArgumentParser(description="Distillation Method for Combining two styles")
    parser.add_argument('--datapath', type=str, required=True, help="Absolute path to dataset")
    parser.add_argument('--model1', type=str, required=True, help="Absolute path to pre-trained model 1")
    parser.add_argument('--model2', type=str, required=True, help="Absolute path to pre-trained model 2")
    parser.add_argument('--save-dir', type=str, default='models', help="Absolute path to save new model")
    parser.add_argument('--log-dir', type=str, default='log_dir', help="Directory for saving logs")
    parser.add_argument('--log-interval', type=int, default=100, help="Directory for saving logs")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help="Directory for saving checkpoints")
    parser.add_argument('--checkpoint-interval', type=int, default=2000, help="Checkpoint Intervals")
    parser.add_argument('--image-size', type=int, default=256, help='Size of your training images')
    parser.add_argument('--batch-size', type=int, default=4, help="batch size")
    parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="number of epochs")
    parser.add_argument('--savename', type=str, default=None, help="name for saved model file")

    args = parser.parse_args()

    check_urself(args)
    train(args)


if __name__ == "__main__":
    main()

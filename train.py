import argparse
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from torch.optim import Adam
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from vgg import VGG16
from transformer import Transformer


def train(args):
    print("Start Time:\t{}".format(time.ctime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(args.datapath, transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mul(255))
                ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    vgg = VGG16().to(device)
    transformer = Transformer(norm='instance', padding='reflect').to(device)
    optimizer = Adam(transformer.parameters(), args.learning_rate)
    mse_loss = torch.nn.MSELoss()

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # allows for multiple styles
    style_images = args.stylepath.split(',')
    num_style = len(style_images)
    styles = load_image(style_images, size=args.style_size)
    styles = [style_transform(style) for style in styles]

    # This needs to be according to n_batch
    # match number of content features to calculate gram matrix losses
    styles = [style.repeat(args.batch_size, 1, 1, 1).to(device) for style in styles]

    # output of vgg is dictionary {"relu1": tensor, "relu2": tensor, ...}
    feature_styles = [vgg(normalize(style)).values() for style in styles]
    gram_style = [gram_matrix(y) for feature_style in feature_styles for y in feature_style]

    flag = True     # used for alternating style images to use
    loss = []
    run_time = time.strftime("%d-%H-%M-%S")

    for epoch_num in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            y_content = x.to(device)
            y_hat = transformer(y_content)

            y_content = normalize(y_content)
            y_hat = normalize(y_hat)

            features_yc = vgg(y_content)
            features_yh = vgg(y_hat)

            content_loss = args.content_weight * mse_loss(features_yh["relu2"], features_yc["relu2"])

            # gram matrix for each transformer network output
            gram_yh = [gram_matrix(feature) for feature in features_yh.values()]

            # entries adjusted for number of styles
            #  <---------style1---------> <-------style2---------->, .... style_n
            # [relu1, relu2, relu3, relu4, relu1, relu2, relu3, ... ]
            gram_yh = [thing for _ in range(num_style) for thing in gram_yh]

            # Calculate style loss
            style_loss = [mse_loss(G_yh, G_style[:n_batch, :, :]).cpu() for G_yh, G_style in zip(gram_yh, gram_style)]

            # Interpolate between two styles
            if args.alpha is not None and num_style == 2:
                alpha_list = [args.alpha for _ in range(4)]             # alpha*first image
                [alpha_list.append(1-args.alpha) for _ in range(4)]     # (1-alpha)*second image
                style_loss = [alpha*loss for loss, alpha in zip(style_loss, alpha_list)]

            # Alternating style image losses for training purposes
            if args.alt is not None:
                if count % (4*args.alt) == 0:
                    flag = not flag
                if flag:
                    style_loss = sum(style_loss[:4])    # first image
                else:
                    style_loss = sum(style_loss[4:])     # second image
            else:
                style_loss = sum(style_loss) / num_style  # both images

            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "[{}/{}]\tTotal: {:.2f}\tStyle: {:.2f}\tContent: {:.2f}".format(
                    count,
                    len(train_dataset),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_content_loss/(batch_id+1),
                )
                print(mesg, end='\r')

                loss.append([batch_id+1,
                             agg_content_loss / (batch_id + 1),
                             agg_style_loss / (batch_id + 1),
                             (agg_content_loss + agg_style_loss) / (batch_id + 1)])

            if args.checkpoint_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch_num+1) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()
                save_loss_plot(np.array(loss), args.log_dir + '/train_loss{}.jpg'.format(run_time))

    # save model and parameter log when training is finished
    transformer.eval().cpu()

    if args.savename is None:
        save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.strftime("%d-%H-%M-%S")) + "_" + str(
            args.content_weight) + "_" + str(args.style_weight) + ".model"
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
        f.write("Style Image: {}\n".format(args.stylepath))
        f.write("Content Weight: {}\n".format(args.content_weight))
        f.write("Style Weight: {}\n".format(args.style_weight))
        f.write("Learning Rate: {}\n".format(args.learning_rate))
        if args.alpha is not None:
            f.write("Alpha: {}\n".format(args.alpha))
        if args.alt is not None:
            f.write("Alternation: {} batches".format(args.alt))

    print("\nDone, trained model saved at", save_model_path)


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

    style_images = args.stylepath.split(',')
    for image in style_images:
        if not os.path.exists(image.strip()):
            print("You better change something. I couldn't find your style image")
    return


def normalize(tens):
    # normalize images according to ImageNet mean and std
    mean = tens.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = tens.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (tens.div_(255.0)-mean)/std


def gram_matrix(tens):
    b, c, h, w = tens.size()
    features = tens.view(b, c, h * w)
    gram = features.bmm(features.transpose(1, 2))
    gram /= c*h*w
    return gram


def load_image(filename, size=None, scale=None):
    images = []
    for file in filename:
        img = Image.open(file.strip())
        if size is not None:
            img = img.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        images.append(img)
    return images


def save_loss_plot(loss, path):
    iters = loss[:, 0]
    total = loss[:, 3]
    content = loss[:, 1]
    style = loss[:, 2]

    plt.plot(iters, total, 'k-', label='Total Loss', linewidth=2)
    plt.plot(iters, content, 'r-', label='Content Loss', linewidth=2)
    plt.plot(iters, style, 'b-', label='Style Loss', linewidth=2)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Fast Neural Style")
    parser.add_argument('--datapath', type=str, required=True, help="Absolute path to Dataset for training")
    parser.add_argument('--stylepath', type=str, required=True, help="Absolute path to Style Image(s)")
    parser.add_argument('--save-dir', type=str, default='models', help="directory to store trained model")
    parser.add_argument('--content-weight', type=float, default=1e5, help="Content loss weighting")
    parser.add_argument('--style-weight', type=float, default=1e10, help="Style loss weighting")
    parser.add_argument('--checkpoint-interval', type=int, default=10000, help="Interval for saving model checkpoints")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help="Directory for saving checkpoints")
    parser.add_argument('--log-interval', type=int, default=100, help="Interval for displaying loss")
    parser.add_argument('--log-dir', type=str, default='log_dir', help="Directory for saving logs")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256, help='size of content images, COCO is 256x256')
    parser.add_argument('--style-size', type=int, default=None, help="resize style image before training")
    parser.add_argument('--alpha', type=float, default=None, help="Weighting for each style-image (alpha) & (1-alpha). "
                                                                  "Must be only two styles")
    parser.add_argument('--alt', type=int, default=None, help="Number of batches for alternating style losses")
    parser.add_argument('--savename', type=str, default=None, help="This is the name your model will be saved as")
    args = parser.parse_args()

    check_urself(args)
    train(args)


if __name__ == "__main__":
    main()



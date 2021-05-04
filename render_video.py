import time
import numpy as np
import cv2
import torch
import argparse
from torchvision import transforms
from transformer import Transformer
import ffmpeg
import os


def main(args):
    content_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_model = Transformer()
    state_dict = torch.load(args.model)
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    im_list = []

    cap = cv2.VideoCapture(args.input)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content_image = content_transform(frame)
            content_image = content_image.unsqueeze(0).to(device)
            output = style_model(content_image).cpu()
            output = output.squeeze(0).detach().clamp(0, 255).numpy()
            output = np.moveaxis(output, 0, -1).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            output = cv2.resize(output, (int(args.resize * output.shape[1]), int(args.resize * output.shape[0])))
            im_list.append(output)

        else:
            print("Video Finished")
            break

    cap.release()
    cv2.destroyAllWindows()




    # referenced from Kyle Mcdonald on https://github.com/kkroening/ffmpeg-python/issues/246
    framerate = 60
    vcodec = 'libx264'
    fn = args.save_name
    images = im_list

    if not isinstance(images, np.ndarray):
        images = np.asarray(images)

    n, height, width, channels = images.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
        .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
        .global_args('-loglevel', 'warning')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )
    process.stdin.close()
    process.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to video to be rendered")
    parser.add_argument("--model", type=str, help="path to style model used for the rendering")
    parser.add_argument("--save-name", type=str, help="name used for saving rendered video")
    parser.add_argument("--resize", default=1.0, type=float, help="rescale image size (keeps aspect ratio)")
    args = parser.parse_args()

    main(args)

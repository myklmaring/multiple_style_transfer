import argparse
import numpy as np
from transformer import Transformer
from PIL import Image
import torch
import torchvision.transforms as transforms

def render(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        content_image = Image.open(args.content_path)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        model = Transformer()
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)
        model.to(device)

        image = model(content_image).cpu()
        save_image(args.output_path, image[0])


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = np.moveaxis(img, 0, -1).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)


def main():
    parser = argparse.ArgumentParser(description="Stylize and Image")
    parser.add_argument('--content-path', type=str, required=True, help="path to content image")
    parser.add_argument('--output-path', type=str, required=True, help="path for output image")
    parser.add_argument('--model-path', type=str, required=True, help="path to trained model")
    args = parser.parse_args()

    render(args)


if __name__ == "__main__":
    main()

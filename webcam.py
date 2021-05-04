import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from transformer import Transformer

#__________________________________________#
model = "models/candy.model"
#__________________________________________#


content_transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_model = Transformer()
state_dict = torch.load(model)
style_model.load_state_dict(state_dict)
style_model.to(device)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    print("Video Capture Failed")

while rval:
    proc_start = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    content_image = content_transform(frame)
    content_image = content_image.unsqueeze(0).to(device)
    style_start = time.time()
    output = style_model(content_image)
    style_end = time.time()
    output = output.squeeze(0).cpu().detach().numpy()
    output = np.moveaxis(output, 0, -1)
    output = (output-output.min())*(1/(output.max()-output.min()))  # range(0,1)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    proc_end = time.time()

    # print("style:\t\t", style_end-style_start, "seconds")
    # print("processing:\t", proc_end-proc_start, "seconds")

    cv2.imshow("preview", output)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")


#!/bin/bash

# Loss method
DATA="/home/michael/Documents/datasets/COCO/"
STYLE="images/style-images/candy.jpg, images/style-images/rain-princess.jpg"

python train.py --datapath "${DATA}" --stylepath "${STYLE}" --epochs 1 \
        --log-interval 100 --checkpoint-interval 10000 --savename "van_munch.model"

## Distillation Method
#model1="${base}/models/monet.model"
#model2="${base}/models/picasso.model"
#datapath="/home/michael/Documents/datasets/COCO/train2017"
#
#python distillation.py --datapath ${datapath} --model1 ${model1} --model2 ${model2} --savename "monasso-distillation.model"
#!/bin/bash

# Loss method
DATA="/home/michael/Documents/datasets/COCO/"
STYLE="images/style-images/candy.jpg, images/style-images/rain-princess.jpg"

python train.py --datapath "${DATA}" \
                --stylepath "${STYLE}" \
                --epochs 1 \
                --log-interval 100 \
                --checkpoint-interval 10000 \
                --savename "van_munch.model"

## Distillation Method
#model1="models/monet.model"
#model2="models/picasso.model"
#DATA="/home/michael/Documents/datasets/COCO/"
#
#python distillation.py --datapath "${DATA}" \
#                       --model1 "${model1}" \
#                       --model2 "${model2}" \
#                       --savename "monasso-distillation.model"
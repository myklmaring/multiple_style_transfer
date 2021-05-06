#!/bin/bash

INPUT="videos/content/champagne.mp4"
MODEL="models/candy.model"
SAVENAME="videos/renders/champagne_candy.mp4"
RESIZE=0.5

python render_video.py --input ${INPUT} --model ${MODEL} --save-name ${SAVENAME} --resize ${RESIZE}
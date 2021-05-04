#!/bin/bash

INPUT="videos/content/champagne.mp4"
MODEL="models/mosaic.model"
SAVENAME="videos/renders/champagne_mosaic.mp4"
RESIZE=0.3

python render_video.py --input ${INPUT} --model ${MODEL} --save-name ${SAVENAME} --resize ${RESIZE}
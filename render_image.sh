#!/bin/bash

base="/home/michael/Documents/git-repos/projecteecs442/original/"
contentpath="${base}/images/content-images/amber.jpg"
modelpath="${base}/models/monet.model"
outputpath="${base}/images/output-images/amber-monet.jpg"

python render.py --content-path ${contentpath} --model-path ${modelpath} \
        --output-path ${outputpath}

modelpath="${base}/models/monasso-distillation.model"
outputpath="${base}/images/output-images/amber-monasso-distillation.jpg"

python render.py --content-path ${contentpath} --model-path ${modelpath} \
        --output-path ${outputpath}

modelpath="${base}/models/monasso-loss.model"
outputpath="${base}/images/output-images/amber-monasso-loss.jpg"

python render.py --content-path ${contentpath} --model-path ${modelpath} \
        --output-path ${outputpath}

modelpath="${base}/models/picasso.model"
outputpath="${base}/images/output-images/amber-picasso.jpg"

python render.py --content-path ${contentpath} --model-path ${modelpath} \
        --output-path ${outputpath}
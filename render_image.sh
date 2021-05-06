#!/bin/bash

contentpath="images/content-images/amber.jpg"
modelpath="models/candy.model"
outputpath="images/output-images/amber-candy1.jpg"

python render.py --content-path "${contentpath}" \
                 --model-path "${modelpath}" \
                 --output-path "${outputpath}"
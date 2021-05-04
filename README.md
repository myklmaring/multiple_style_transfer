# multiple_style_transfer

This is an original python implementation of real-time style transfer [[1]](#1).  This implementation allows for the combination of two or more styles during the style transfer.  The first method I used to accomplish this was summing the losses from each style image during training.  The alternative method I used to accomplish multiple style transfer was using distillation [[2]](#2) with pretrained single-style style transfer transformers.

## References
<a id="1">[1]</a> 
Justin Johnson, Alexandre Alahi, and Li Fei-Fei (2016). 
Perceptual Losses for Real-Time Style Transfer
and Super-Resolution

<a id="2">[2]</a> 
Geoffrey Hinton and Oriol Vinyals and Jeff Dean (2015)
Distilling the Knowledge in a Neural Network

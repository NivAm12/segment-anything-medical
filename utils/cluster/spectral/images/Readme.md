# Spectral clustering  
Apply spectral clustering to an image according to https://arxiv.org/abs/2205.07839  
With some mild changes can cluster other modelity of data not only images

## Please run spectral.py  
Requires 3 arguments:

image_dir: Input image directory

K: Number of eigan values to use (Also number of cluster if adaptive is False)

adaptive: if true; will chose number of cluster using eigengap.

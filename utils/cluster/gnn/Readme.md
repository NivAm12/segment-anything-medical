# GNN segmentation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EAxOuLRG7Cmwi1UI6-M-pknLM2rAfYMi?usp=sharing)
![](images/figs/fig.png)
![](images/figs/sem_seg.png)

## How to use?

### segment.py
Will divide an input image to K clusters and will show results on the screen.

### co_segment.py
Will apply co segmentation to a set of images (Works only for k==2)

### segment_dataset.py
Non-interactive segment.py, will go thorough an entire dataset and save a segmentation map
or a bounding box of the main object 

## Examples:
![](images/figs/reflact.png)![](images/figs/2.png)
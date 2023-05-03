# Clustering methods

## Spectral clustering
Segmentation of images, But can apply to different data modalities.  
From: https://arxiv.org/abs/2205.07839  
Implemented by: @Amit Aflalo

## GNN clustering
Segmentation of images, But can apply to different data modalities.    
Based on optimization of N-cut problem using GNN (graph neural network)  
Author: @Amit Aflalo 

## How to run?
You can use the following singularity container: ```/home/projects/yonina/SAMPL_training/sif_files/gnn.sif```

Commands to run singularity environment:  
Ask for a machine:  
```bsub -q waic-short -gpu num=1:j_exclusive=yes -R rusage[mem=16192] -R affinity[thread*8] -m waic_dgx_hosts -R "hname!=dgxws01 && hname!=dgxws02" -Is /bin/bash```

Load Singularity:  
```module load Singularity```

Open singularity container:  
```singularity shell --nv --bind /home/hsd/, /home/projects/yonina/SAMPL_training/sif_files/gnn.sif```

Optional, open PyCharm:  
```export DISPLAY=<your ip adress> ```  
```/pycharm-community-2021.2.2/bin/pycharm.sh &```

Don't forget to choose the interpreter that is inside the singularity container.  
interpreter path: ```/opt/conda/bin/python```
# ALASKA2 Image Steganalysis

Source code of solution for [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis) competition.

## Solution 

Key points: 
* Efficientnets
* DDP training with SyncBN and Apex mixed precision
* AdamW with cosine annealing
* EMA Model
* Bitmix

## Quick setup and start 

### Requirements 

*  Nvidia drivers, CUDA >= 10.2, cuDNN >= 7
*  [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 

The provided dockerfile is supplied to build image with cuda support and cudnn.


### Preparations 

* Clone the repo, build docker image. 
    ```bash
    git clone https://github.com/lRomul/argus-alaska.git
    cd argus-alaska
    make build
    ```

* Download and extract [dataset](https://www.kaggle.com/c/alaska2-image-steganalysis/data) to `data` folder.

### Run

* Run docker container 
```bash
make
```

* Create folds split and extract quality of images 
```bash
python make_folds.py
python make_quality_json.py
```

* Train model
```bash
python train.py --experiment train_001
```

* Predict test and make submission 
```bash
python predict.py --experiment train_001
```

* Train on 4 GPUs with distributed data parallel 
```bash
./distributed_train.sh 4 --experiment ddp_train_001
```

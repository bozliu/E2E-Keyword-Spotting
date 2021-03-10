# E2E-Keyword-Spotting

Joint End to End Approaches to Improving Far-field Wake-up Keyword Detection

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


1. Install dependent packages

    ```bash
    cd E2E-Keyword-Spotting
    pip install -r requirements.txt
    ```
2. Or use conda 
    ```bash
    cd E2E-Keyword-Spotting
    conda env create -f environment.yaml
    ```

## :turtle: Dataset Preparation

#### How to Use
Dataset is from [Google Speech Command](./basicsr/data/paired_image_dataset.py) 
* Directly read disk data.
```yaml
type: PairedImageDataset
dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
io_backend:
  type: disk
```
* Data Pre-processing (Has already been done)
1. According to the file, dataset has already been splited into three folders, train, test, and valid. 
1. The splited [Google Speech Command dataset](https://drive.google.com/file/d/1InqR8n7l5Qj6voJREpcjHYWHVTKG-BbB/view?usp=sharing) is saved on Google Drive folder. 
    
## :computer: Train and Test
### Training commands
- **Single GPU Training**: 
```
python train.py
```
- **Distributed Training**: 
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
### Test commands
```
python test.py 
```


## Result

Result can be viewed on [wandb](https://wandb.ai/bozliu/google_speech_command?workspace=user-bozliu)

### Model Parameters 
![image](images/model_parameters.png)

###

## Files Description
├── assets
├── basicsr
    ├── data
    ├── metrics
    ├── models
    └── utils
├── basicsr.egg-info
├── build
├── colab
├── datasets
    ├── DIV2K
    ├── LR
├── docs
├── experiments
├── LICENSE
├── options
    ├── test
    │   ├── DUF
    │   ├── EDSR
    │   ├── EDVR
    │   ├── ESRGAN
    │   ├── RCAN
    │   ├── SRResNet_SRGAN
    │   └── TOF
    └── train
        ├── EDSR
        ├── EDVR
        ├── ESRGAN
        ├── RCAN
        ├── SRResNet_SRGAN
        └── StyleGAN
├── results
├── scripts
├── tb_logger
├── tests
├── test_scripts
├── tmp
└── wandb

* *./basicsr/data* parpares diffferent form of dataset and data augumentation 
* *./basicsr/models* saves diffferent network architecture 
* *./datasets/DIV2K* : Mini-DIV2K dataset for training and validation 
* *./datasets/LR* : Test Set
* *./results*: output of generated upscaled images from testset
* *./options* : includes configuration for training and testing of different models

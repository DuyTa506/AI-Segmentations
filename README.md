# AI End Course Project
### Table of contents
1. [Documentation](#documentation)
2. [Available Features](#feature)
3. [Installation](#installation)
4. [Train](#train)
5. [Inference](#inference)
6. [Logs and Visualization](#logs)

<a name = "documentation" ></a>
### Documentation
Unet is the most well-known segmentation network for segmentation. It was first designed to adapt the biomedical image domain. In this project, we will test the performance of UNet with an urban city datasets - Cityscapes
</br>
All documents related to this repo can be found here:
- [Unet Pytorch](https://github.com/milesial/Pytorch-UNet)
- [Paper](https://arxiv.org/abs/1505.04597)

<a name = "feature" ></a>
### Available Features
- [x] Baseline Code
- [ ] Clean and CLI interaction
<a name = "installation" ></a>
### Installation
```
pip install -r requirements.txt
```

<a name = "train" ></a>
### Train
1. Prepare your dataset
    - First, move into <b>.notebook directory </b> and find  <b>data_prepare.ipynb</b> notebook, run first 4 cells to download the Cityscapes dataset. You'll need to have an account. Sign up for a new account at [Cityscapes's Homepage](https://www.cityscapes-dataset.com/)
    - After this, it will automatically download and split the datasets into train, val and test subsets. The dataset's size is about 11GB.
   
2. This project will be updated for CLI interaction, but now, if you want to change the model architechture, hyperparameters and so on ... Please modify it in python code <b>train.py<b>
3. Run
    - Start training from scratch:
      
        ```
        python train.py
        ```
     - This project support several model architechture:
         - FCN , UNet , UNet with ResNet encoder , PSPNet, DeepLabv3.
       
<a name = "inference" ></a>
### Inference

```cmd
usage: test.py

```

<a name = "logs" ></a>
### Logs and Visualization
The logs during the training will be stored, and you can visualize it using TensorBoard by running this command:
```
tensorboard --logdir ~/saved/<project_name>

# specify a port 8080
tensorboard --logdir ~/saved/<project_name> --port 8080
```


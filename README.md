# :zap: Understand the workflow of image segmentation with Unet and CityScapes
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
    - First, move into <b>.notebook directory </b> and find  <b>.data_prepare.ipynb</b> notebook, run first 4 cells to download the Cityscapes dataset. You'll need to have an account. Sign up for a new account at [Cityscapes's Homepage](https://www.cityscapes-dataset.com/)
    - After this, it will automatically download and split the datasets into train, val and test subsets. The dataset's size is about 11GB.
   
2. Configure the [config.toml](config.toml) file: Pay attention to the <b>pretrained_path</b> argument, it loads "facebook/wav2vec2-base" pre-trained model from Facebook by default. Change it to the pretrained models from phase 1 if need
3. Run
    - Start training from scratch:
        ```
        python train.py -c config.toml
        ```
    - Resume:
        ```
        python train.py -c config.toml -r
        ```
    - Load specific model and start training:
        ```
        python train.py -c config.toml -p path/to/your/model.tar
        ```

<a name = "inference" ></a>
### Inference
We provide an inference script that can transcribe a given audio file or even a list of audio files. Please take a look at the arguments below, especially the ```-f TEST_FILEPATH``` and the ```-s HUGGINGFACE_FOLDER``` arguments:
```cmd
usage: inference.py [-h] -f TEST_FILEPATH [-s HUGGINGFACE_FOLDER] [-m MODEL_PATH] [-d DEVICE_ID]

ASR INFERENCE ARGS

optional arguments:
  -h, --help            show this help message and exit
  -f TEST_FILEPATH, --test_filepath TEST_FILEPATH
                        It can be either the path to your audio file (.wav, .mp3) or a text file (.txt) containing a list of audio file paths.
  -s HUGGINGFACE_FOLDER, --huggingface_folder HUGGINGFACE_FOLDER
                        The folder where you stored the huggingface files. Check the <local_dir> argument of [huggingface.args] in config.toml. Default
                        value: "huggingface-hub".
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the model (.tar file) in saved/<project_name>/checkpoints. If not provided, default uses the pytorch_model.bin in the
                        <HUGGINGFACE_FOLDER>
  -d DEVICE_ID, --device_id DEVICE_ID
                        The device you want to test your model on if CUDA is available. Otherwise, CPU is used. Default value: 0
```

Transcribe an audio file:
```cmd
python inference.py \
    -f path/to/your/audio/file.wav(.mp3) \
    -s huggingface-hub

# output example:
>>> transcript: Hello World 
```

Transcribe a list of audio files. Check the input file [test.txt](examples/inference_data_examples/test.txt) and the output file [transcript_test.txt](examples/inference_data_examples/transcript_test.txt) (which will be stored in the same folder as the input file):
```cmd
python inference.py \
    -f path/to/your/test.txt \
    -s huggingface-hub
```


<a name = "logs" ></a>
### Logs and Visualization
The logs during the training will be stored, and you can visualize it using TensorBoard by running this command:
```
# specify the <project_name> in config.json
tensorboard --logdir ~/saved/<project_name>

# specify a port 8080
tensorboard --logdir ~/saved/<project_name> --port 8080
```
![tensorboard](examples/images/tensorboard.jpeg)


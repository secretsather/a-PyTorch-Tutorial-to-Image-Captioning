This is a fork of [Sagar Vinodababu's](https://github.com/sgrvinod) **[PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)**.

Please check out his [original work](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) and appreciate the effort put into this. He has provided a great tutorial and explantion of his implementation of the [_Show, Attend, and Tell_](https://arxiv.org/abs/1502.03044) paper.

# Contents

[***Why another fork***](#why-another-fork)

[***Workflow***](#workflow)

[***Summary***](#summary)

[***Changelog***](#changelog)

# Why another fork

This fork was created to preserve this code, while addressing compatibility into future versions of PyTorch and other dependent libraries. Moreover, it is this author's intent to commit many quality of life features as it relates to managing network settings and datasets, while allowing for easy implementation using custom datasets. The original [README.md](docs/README_ORIGINAL.md) has been preserved for information; however, the commands given in the original should not be assumed to work. 

# Workflow

The main steps for custom images to a caption generator are as follows: 
1. [Setup environment](#setup-environment)
2. [Collect images & manually create captions](#collect-images-create-captions)
3. [Create a dataset JSON that mirrors the COCO Captions dataset](#create-dataset-json)
4. [Setup dataset for compatibility with original repository](#setup-dataset)
5. [Train custom network](#training)
6. Evaluate trained network (optional)
7. [Generate caption predictions using untrained images](#generate-captions)

## Setup Environment

While this isn't meant to be a guide for setting up a working python environment, I will list the versions of the libraries that are confirmed to work on this fork below. This was created on a Win 11 machine with the following libraries/dependencies:
* PyTorch 1.11.0 + cu113
* Argparse
* (todo - populate this)

## Collect Images Create Captions

This fork expects a dataset in the same format as [MS COCO 2014 caption validation set](https://github.com/tylin/coco-caption). That said, if you're using a pre-made dataset, you may skip to the next section. When using a custom dataset, this fork expects a folder of images, each with a .txt file of the same name in the same folder with the caption. This script currently only supports one caption per image, but modifying it shouldn't be too hard if you need that to suit your dataset. 

## Create Dataset JSON

A script has been provided, [1_dataset_to_json.py](1_dataset_to_json.py) which will create a JSON file compatible with the training for this and the original repository, which one may also use for other repositories expecting the same format. The usage of this script is as follows:

To run the script, use the below command replacing the <ExperimentName> and <AbsoluteImagePath> with values of your choosing. 
```python
python 1_dataset_to_json.py <ExperimentName> <AbsoluteImagePath>
```
You may use a --BYOdataset flag if you're training off a dataset that is already compatible (e.g. MS Coco Captions). In this instance, you will still need to specify the absolute path of your images, as well as the <ExperimentName>, as the script makes a folder structure and settings file for your experiment. 

This script, when first run, will create a directory structure within the root of the repository formatted as below:
```
> root
  > networks
    > <ExperimentName>
      > dataset
      > img_out
      > txt_out
      > checkpoints
```

Two files will be placed in the <ExperimentName> folder:
* <ExperimentName>.json - The JSON file for your dataset that is formatted similarly to MSCOCO Captions
* <ExperimentName>_settings.json - The settings for your network

If you're bringing your own dataset, you do not need to copy any images, only provide the absolute path to them. You should however, make a copy of the .json file and put it in the root of <ExperimentName> while renaming the file to <ExperimentName>.json (i.e. it should have the same name as the directory it is in)

If you're using your own data, you can simply provide a name and the absolute image path (e.g. c:\images). You may edit the <ExperimentName>_settings.json file to suit your network. This will create a JSON file with a 80,10,10 -> train,test,validation split. This number can be changed by editing the [1_dataset_to_json.py](1_dataset_to_json.py) file

## Setup Dataset

Next, run the 2_create_input_files.py file as the following: 
```
python 2_create_input_files.py <ExperimentName>
```

This file is from the author's orignal work, modified to use the values in the <ExperimentName>_settings.json file. It will make a copy of ALL your images inside the <ExperimentName>/dataset directory, along with the captions. Use with caution, as you may fill your hard drive if working with extremely large datasets since it makes a copy of everything. The h5py file allows python to use the data directly from the hard drive as if it's reading it from memory. This was likely done to reduce processing time and memory footprint, especially if you're using the GPU. The original author also resizes the images to 256x256 in this procedure, so the images being fed to your network may be smaller than their originals. This is okay, since you're only generating captions. 

After a successful run, you should see "Reading TRAIN images and captions, storing to file..." for the train, val, and test images. 

## Training

Training can be run using: 
```
python 3_train.py <ExperimentName>
```
The --resume flag will resume training from the best checkpoint, as determined by the BLEU-4 score. The --fineTune flag will unlock some of the layers of the encoder for continued training. The author of the original work suggests running training until no improvement is seen, then re-run training with fine tuning. 

This will run until "epochs" are reached (in the settings file) or if there is no improvement in the BLEU-4 score (also in the settings). The original author describes this in depth. This implementation has not changed that. 

## Evaluation

--This is not implemented yet--

## Generate Captions

For inference, run:
```
python 4_caption.py datasetName
```
The images used for inference can be set in your <ExperimentName>_settings.json file that was created in step 1 under testing>>path as below:
```
    "testing": {
        "path": "d:/temp",
```
You may use the --max_images argument followed by an integer to only make captions for X amount of images. You may use the --txt_only switch to create text captions with no images, otherwise the script will create both text files and images. 

The images will be created in <ExperimentName>/img_out/ and the text files with predicted captions will be in the <ExperimentName>/txt_out.  The text files will be named with the same name as the images they were derived from. 

# Summary

A typical training/inference session would be run with the below commands: 
```
python 1_dataset_to_json.py testNet c:\path_to_images_and_txt_files
(edit ./networks/testNet/testNet_settings.json to suit your data)
python 2_create_input_files.py testNet
python 3_train.py testNet
python 3_train.py testNet --resume --fineTune
python 4_caption.py testNet --max_images 5
```

# Changelog
1. Move settable parameters from individual .py files to JSON file, initialized at 1_dataset_to_json.py
2. Use of PIL.Image in lieu of scipy.misc.imread and scipy.misc.imresize due to depreciation. 

import os
import json
import random
import argparse

def checkMakeDir(d):    
    if not os.path.exists(d):
        os.mkdir(d)
        print(f'{d} created.')

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("datasetName", help="Enter the name of the dataset/experiment")
parser.add_argument("absoluteIMGpath", help="Enter the absolute image path")
parser.add_argument("--BYOdataset", action="store_true", help="Use this switch if your training off a different dataset")
args = parser.parse_args()

#process arguments/globals
args.absoluteIMGpath = args.absoluteIMGpath.replace('\\', '/') #for win users
imgPath = args.absoluteIMGpath
splits = [80, 10, 10] #train, test, val

#create directory structure
dirs = ['./networks', f'./networks/{args.datasetName}', f'./networks/{args.datasetName}/dataset', 
        f'./networks/{args.datasetName}/img_out', f'./networks/{args.datasetName}/checkpoints',
        f'./networks/{args.datasetName}/txt_out']
for d in dirs:
    checkMakeDir(d)
   
jsonFname = f'./networks/{args.datasetName}/{args.datasetName}.json'
settingsjson = f'./networks/{args.datasetName}/{args.datasetName}_settings.json'

if not args.BYOdataset:
    '''CREATE DATASET JSON'''

    imgNames = []
    captions = []
    allFiles = os.listdir(imgPath)
    random.shuffle(allFiles)

    for f in allFiles:
        if f[-3:].lower() in {'jpg', 'png'}:
            if os.path.exists(f'{imgPath}/{f[0:-3]}txt'):
                imgNames.append(f)
                with open(f'{imgPath}/{f[0:-3]}txt') as txtFile:
                    captions.append(txtFile.read())   
    del allFiles

    assert len(imgNames) == len(captions)

    splitIndex = [int((splits[0] / 100) * len(imgNames)), int((splits[1] / 100) * len(imgNames))]
    #splitIndex.append(len(imgNames) - (splitIndex[0] + splitIndex[1]))

    for i in range(len(imgNames)):
        splits.append('train')
        if i >= splitIndex[0]:
            splits[-1] = 'test'
            if i >= splitIndex[0] + splitIndex[1]:
                splits[-1] = 'val'

    images = []

    for i in range(len(imgNames)):
        #make tokens for text
        tokens = captions[i].split(' ')
        #put in sentences List
        sentence = {"tokens":tokens, "raw":captions[i], "imgid":i}
        sentences = [sentence]
        #create image dictionary and put in images list
        images.append({"filepath": '', "filename": imgNames[i], "imgid": i, "split": splits[i], 
                       "sentences": sentences, })

    #create dictionary with image list
    jsonData = {"images": images}
    del images

    #write json    
    with open(jsonFname, 'w') as jsonFile:
        json.dump(jsonData, jsonFile)
    
    print(f'{len(imgNames)} image/txt pairs found.')
    del jsonData
    del imgNames
    del captions

'''CREATE JSON SETTINGS FILE'''

dataset = {
    "JSON": jsonFname,
    "PATH": args.absoluteIMGpath,
    "CAPTIONS_PER_IMAGE": 1,
    "MIN_WORD_FREQUENCY": 0,
    "MAX_CAPTION_LENGTH": 50
}
network = {
    "Network_NAME": args.datasetName,
    "start_epoch": 0,
    "epochs": 120,
    "batch_size": 32,
    "workers": 0,           # for data-loading; right now, only 1 works with h5py
    "encoder_lr": 0.0001,   # learning rate for encoder if fine-tuning
    "decoder_lr": 0.0004,   # learning rate for decoder
    "grad_clip": 5.,        # clip gradients at an absolute value of
    "alpha_c": 1.,          # regularization parameter for 'doubly stochastic attention', as in the paper
    "print_freq": 100,      # print training/validation stats every __ batches
    "no_improvement_quit": 10 #Quit after X amount of epochs with no BLEU improvement
}
testing = {
    "path": args.absoluteIMGpath,
    "beam_size": 5,
}

jsonDATA = {"dataset": dataset, "network":network, "testing":testing}

with open(settingsjson, "w") as jsonFile: 
    json.dump(jsonDATA, jsonFile, indent=4)
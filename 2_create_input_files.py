import os
import json
import argparse
from utils import create_input_files

parser = argparse.ArgumentParser()
parser.add_argument("datasetName", help="Enter the name of the dataset/experiment")
args = parser.parse_args()

with open(f'./networks/{args.datasetName}/{args.datasetName}_settings.json', 'r') as jsonFile:
    setDat = json.load(jsonFile)    

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset=args.datasetName,
                       karpathy_json_path=setDat['dataset']['JSON'],
                       image_folder=setDat['dataset']['PATH'],
                       captions_per_image=setDat['dataset']['CAPTIONS_PER_IMAGE'],
                       min_word_freq=setDat['dataset']['MIN_WORD_FREQUENCY'],
                       output_folder=f'./networks/{args.datasetName}/dataset',
                       max_len=setDat['dataset']['MAX_CAPTION_LENGTH'])

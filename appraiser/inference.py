from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import login
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse

class ItemDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        self.item_frame = dataframe
        self.tokenizer = tokenizer
        self.max_length = 512
        
    def __len__(self):
        return len(self.item_frame)

    def __getitem__(self, idx):
        row = self.item_frame.iloc[idx]
        item_info = row["item_text"]
        encoded_input = self.tokenizer(item_info, return_tensors='pt', max_length = self.max_length, padding='max_length')
        input_ids = encoded_input['input_ids']
        masks = encoded_input['attention_mask']
        price = row["normalized_price"]

        return {'input_ids':input_ids, 'attention_mask':masks}, price


def inference(checkpointdir, outputdir, tokenizer_name, datasetdir, batch_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(checkpointdir) #ENTER SAVED MODEL CHECKOINT PATH
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token


    test_df = pd.read_csv(datasetdir)
    test_dataset = ItemDataset(test_df, tokenizer)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    del(test_df)
    output_list = []
    with torch.no_grad():
        model.eval()
        for _, (data, target) in tqdm(enumerate(test_data_loader)):
            target = target.to(device)
            data['input_ids'] = data['input_ids'].squeeze()
            data['attention_mask'] = data['attention_mask'].squeeze()
            output = model(**data)
            output_list = output_list + output.tolist()

    output_dict = {"model_predicted": output_list}
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(outputdir, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_dir', dest="checkpointdir") #xxx/xx.pt
    parser.add_argument('-o', '--output_dir', dest="outputdir") # xxx/xx.csv
    parser.add_argument('-t', '--tokenizer', dest="tokenizer_name")
    parser.add_argument('-d', '--dataset_dir', dest="datasetdir")
    parser.add_argument('-b', '--batch_size', dest="batch_size")
    args = parser.parse_args()
    inference(args.checkpointdir, args.outputdir, args.tokenizer, args.datasetdir, args.batch_size)

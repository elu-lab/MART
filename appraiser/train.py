from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os
import gc
import argparse

MIN_VALID_LOSS = np.inf

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


class Appraiser(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = nn.DataParallel(AutoModelForCausalLM.from_pretrained(base_model_name), device_ids=[0,1])
        for param in self.base_model.parameters(): # Freeze base model
            param.requires_grad = False

        self.fc_1 = nn.Linear(2048, 512) # for Llama-3.2-1b
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128,32)
        self.fc_4 = nn.Linear(32,1)
        self.activation = nn.GELU()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.base_model(input_ids.to(self.device), 
                            attention_mask = attention_mask.to(self.device), output_hidden_states=True).hidden_states[-1]
        
        last_hidden_state_mean = last_hidden_state.mean(dim=1)
        y = self.activation(self.fc_1(last_hidden_state_mean))  
        y = self.activation(self.fc_2(y))
        y = self.activation(self.fc_3(y))
        y = self.fc_4(y)
        return y.squeeze(-1)


def train(datadir, base_model_name, tokenizer_name, batch_size, epoch, lr, random_seed, checkpointdir):
    global MIN_VALID_LOSS
    seed = random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(datadir+"/train.csv")
    valid_df = pd.read_csv(datadir+"/valid.csv")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    batch_size = batch_size
    train_dataset, valid_dataset = ItemDataset(train_df, tokenizer), ItemDataset(valid_df, tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    del train_df
    del valid_df
    del test_df
    gc.collect()

    model = Appraiser(base_model_name)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.zero_grad()

    for _ in epoch:
        model.train()
        for _, (data, target) in enumerate(tqdm(train_data_loader)):
            optimizer.zero_grad()
            target = target.to(device)
            data['input_ids'] = data['input_ids'].squeeze()
            data['attention_mask'] = data['attention_mask'].squeeze()
            output = model(**data)
            loss = loss_fn(output.float(), target.float()).to(device)
            loss.backward()
            optimizer.step()
        
        valid_loss = evaluation(model, device, valid_data_loader, loss_fn)
        if valid_loss < MIN_VALID_LOSS:
            torch.save(model, checkpointdir+"/appraiser.pt")
            MIN_VALID_LOSS = valid_loss


def evaluation(model, device, valid_data_loader, loss_fn):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(valid_data_loader):
            target = target.to(device)
            data['input_ids'] = data['input_ids'].squeeze()
            data['attention_mask'] = data['attention_mask'].squeeze()
            output = model(**data)
            loss = loss_fn(output.float(), target.float()).to(device)
            total_loss+=loss

    return total_loss / len(valid_data_loader)


if __name__ == "__main__":
    ### huggingface token required to use llama3 from huggingface
    login("YOUR HUGGINGFACE TOKEN")


    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", dest="datadir")
    parser.add_argument("-b", "--base_model_name", dest="base_model_name")
    parser.add_argument("-t", "--tokenizer_name", dest="tokenizer_name")
    parser.add_argument("-bs", "--batch_size", dest="batch_size")
    parser.add_argument("-e", "--epoch", dest="epoch")
    parser.add_argument("-lr", "--learning_rate", dest="lr")
    parser.add_argument("-s", "--random_seed", dest="random_seed")
    parser.add_argument("o", "--checkpointdir", dest="checkpointdir")    
    args = parser.parse_args()

    train(
        args.datadir,
        args.base_model_name,
        args.tokenizer_name,
        args.batch_size,
        args.epoch,
        args.lr,
        args.random_seed,
        args.checkpointdir)
import pandas as pd
import random
from typing import Literal
from openai import OpenAI
import os.path
import time
from datetime import datetime
from tqdm import tqdm
import json
import argparse


def change_view_point(conversation: list):
    """
    Example:
        Input:
            conversation = [{"role": "system", "content": "This is system prompt"}, {"role": "user", "content": "Hello"}, {"role": "assistant": "Hi"}]
        Output:
            conversation = [{"role": "system", "content": "This is system prompt"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Hi"}]

    """
    for conv in conversation:
        if conv["role"] == "assistant": conv["role"] = "user"
        elif conv["role"] == "user": conv["role"] = "assistant"
    return conversation


def merchant_strategy(conversation, item_name, item_info, item_price):
    api_key="YOUR OPENROUTER API KEY"
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",api_key=api_key)

    system_prompt = f"""You are a merchant good at negotiating in the game.
**A game player is trying to buy {item_name}.**
The appropriate price you think is {item_price} golds.
------
ITEM INFORMATION
{item_info}
------
Here are some persuasive strategies you can use while negotiating.
1. Liking: This involves strengthening connections with people through similarities or compliments. By finding common ground and offering praise that can spark a sense of liking, you can build relationships with others and persuade them effectively.
2. Reciprocity: People by their nature feel obliged to return a favor and to pay back others. Thus, when a persuasive request is made by a person the receiver feels indebted to, the receiver is more inclined to adhere to the request.
3. Social Proof: We often observe the behaviors of others to help us make decisions. This is because a large majority of individuals are imitators rather than initiators, and therefore make decisions only after observing the behaviors and consequences on those around them.
4. Consistency: People, by their nature, strive to be consistent with previous or reported behavior to avoid cognitive dissonance.
5. Authority: People defer to experts. Therefore, individuals are more likely to comply with a request when it is made by a person or people they perceived as possessing high levels of knowledge, wisdom, or power.
6. Scarcity: People tend to place more value on things that are in short supply. This is due to the popular belief that less available options are of higher quality.
------
Additionally, following values are key factors that affect game players' decision-making when they purchase items.
1. Enjoyment value: Players can enjoy the game more, find it more exciting, and feel happier.
2. Character competency value: Players can quickly increase their game level, gain more points than before, and enhance their powers.
3. Visual authority value: Players can adorn their game characters to be more stylish, improve their appearance, attract more attention, and make a better impression on others.
4. Monetary value: Game items are worth more than they cost, offer good value for the price, and are reasonably priced.
------
Generate your next utterance's strategy in gerund phase.
Your strategy is not shown to the game player.
Just generate the strategy, not explicit utterance.
Your strategy must be less than 20 words."""
    
    strategy_ask_prompt = f"""What is your negotiative strategy for your next utterance?"""
    
    merchant_strategy = None
    conversation.insert(0, {"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": strategy_ask_prompt})
    while True:
        try:
            time.sleep(2)
            chat_completion = client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages = conversation
            )
            if len(chat_completion.choices[0].message.content) < 2: continue
            elif '\\\\' in chat_completion.choices[0].message.content: continue
            else: 
                merchant_strategy = chat_completion.choices[0].message.content
                break
        except Exception as e:
            print(f"{datetime.now()}: Open Router Error")
            time.sleep(600)
    
    conversation.pop(0)
    conversation.pop(-1)
    return merchant_strategy

    

def merchant_turn(conversation, item_name, item_info, item_price):
    api_key="YOUR OPENROUTER API KEY"
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",api_key=api_key)

    system_prompt = f"""You are a merchant selling items in a fantasy game.
**A game player is trying to buy {item_name}.**
The appropriate price you think is {item_price} golds.
------
ITEM INFORMATION
{item_info}
------
Here are some persuasive strategies you can use while negotiating.
1. Liking: This involves strengthening connections with people through similarities or compliments. By finding common ground and offering praise that can spark a sense of liking, you can build relationships with others and persuade them effectively.
2. Reciprocity: People by their nature feel obliged to return a favor and to pay back others. Thus, when a persuasive request is made by a person the receiver feels indebted to, the receiver is more inclined to adhere to the request.
3. Social Proof: We often observe the behaviors of others to help us make decisions. This is because a large majority of individuals are imitators rather than initiators, and therefore make decisions only after observing the behaviors and consequences on those around them.
4. Consistency: People, by their nature, strive to be consistent with previous or reported behavior to avoid cognitive dissonance.
5. Authority: People defer to experts. Therefore, individuals are more likely to comply with a request when it is made by a person or people they perceived as possessing high levels of knowledge, wisdom, or power.
6. Scarcity: People tend to place more value on things that are in short supply. This is due to the popular belief that less available options are of higher quality.
------
Additionally, following values are key factors that affect game players' decision-making when they purchase items.
1. Enjoyment value: Players can enjoy the game more, find it more exciting, and feel happier.
2. Character competency value: Players can quickly increase their game level, gain more points than before, and enhance their powers.
3. Visual authority value: Players can adorn their game characters to be more stylish, improve their appearance, attract more attention, and make a better impression on others.
4. Monetary value: Game items are worth more than they cost, offer good value for the price, and are reasonably priced.
------
You are negotiating with the game player. Generate your next utterance in no more than two sentences without any prefix."""
    
    merchant_utterance = None
    conversation.insert(0, {"role": "system", "content": system_prompt})

    while True:
        try:
            time.sleep(2)
            chat_completion = client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages = conversation
            )
            if len(chat_completion.choices[0].message.content) < 2: continue
            elif '\\\\' in chat_completion.choices[0].message.content: continue
            else: 
                merchant_utterance = chat_completion.choices[0].message.content
                break
        except Exception as e:
            print(f"{datetime.now()}: Open Router Error")
            time.sleep(600)
    
    conversation.append({"role": "assistant", "content": merchant_utterance})
    conversation = conversation[1:]
    return conversation



def player_turn(conversation,item_name, item_info, player_price):
    client = OpenAI(api_key="YOUR OPENAI API KEY")
    system_prompt = f"""You are a game player playing a fantasy game.
**You are in a shop to buy {item_name}.**
The appropriate price you think is {player_price} golds.
------
ITEM INFORMATION
{item_info}
------
You are negotitating with a merchant. Generate your next utterance in no more than two sentences without any prefix.
If you think the conversation is over, generate nothing but 'CONVERSATION OVER'."""
    
    conversation.insert(0, {"role": "system", "content": system_prompt})

    completion = client.chat.completions.create(
        model = "gpt-4o",
        messages = conversation
    )

    conversation.append({"role": "assistant", "content": completion.choices[0].message.content})
    conversation = conversation[1:]
    return conversation
    

def gen_conversation(item_dataset_path, outputdir, max_turn):
    item_df = pd.read_csv(item_dataset_path) #xxx/xxx.csv
    
    for item_idx in tqdm(range(len(item_df)), position=0, desc="item_idx"):
        early_end=False
        item_name, item_info, merchant_price, player_price, item_id = get_item(item_df, item_idx)
        if os.path.exists(f"{outputdir}/{item_id}.csv"):
            print(f'{item_id} exists')
            continue
        if os.path.exists(f"{outputdir}/{item_id}.csv"):
            print(f'{item_id} exists')
            continue
        conversation = [{"role": "user", "content": f"Hello. I'm looking for {item_name}"}]
        speaker_list = ["Game Player"]
        utterance_list = [f"Hello. I'm looking for {item_name}"]


        for _ in range(max_turn):
            print(f'turn {_}')
            #merchant turn
            strategy = merchant_strategy(conversation, item_name, item_info, merchant_price)
            conversation.append({"role": "assistant", "content": f"({strategy})"})
            conversation = merchant_turn(conversation, item_name, item_info, merchant_price)
            speaker_list.append("Merchant")
            utterance_list.append(f"({strategy}) {conversation[-1]['content']}")
            conversation.pop(-2) #remove strategy asking
            
            #player turn
            conversation = change_view_point(conversation)
            conversation = player_turn(conversation, item_name, item_info, player_price)
            speaker_list.append("Game Player")
            utterance_list.append(conversation[-1]["content"])


            if "conversation over" in conversation[-1]["content"].lower():
                gen_json(conversation, item_name, item_info, item_id, merchant_price, player_price, outputdir)
                gen_csv(speaker_list, utterance_list, item_id, outputdir)
                early_end = True
                break
            conversation = change_view_point(conversation)
        if not early_end:
            gen_json(conversation, item_name, item_info, item_id, merchant_price, player_price, outputdir)
            gen_csv(speaker_list, utterance_list, item_id)
    

def gen_csv(speaker_list, utterance_list, item_id, outputdir):
    tmp_dict = {"speaker": speaker_list, "utterance": utterance_list}
    tmp_df = pd.DataFrame(tmp_dict)
    tmp_df.to_csv(f"{outputdir}/{item_id}.csv", index=False)

def gen_json(conversation: list, item_name, item_info, item_id, merchant_price, player_price, outputdir):
    info_dict = {
        "environment": {
            "item_name": item_name,
            "item_info": item_info,
            "item_id": item_id,
            "merchant_price": merchant_price,
            "player_price": player_price        
            },
        "conversation": conversation
    }
    with open(f"{outputdir}/{item_id}.json", "w+") as f: 
        json.dump(info_dict, f, indent=2)


def get_item(df: pd.DataFrame, idx: int) -> tuple:
    """
    Get item from item dataframe by idx
    Returns:
        item_name (str)
        item_info (str)
        item_price (str): price for merchant
        user_price (str): price for game player (1%-10% discount from item_price)
    """

    row = dict(df.iloc[idx])
    item_name = row["item_text"].split("\n")[0]
    item_id = int(row["itemid"])
    item_info = "\n".join(row["item_text"].split("\n")[1:])
    item_price = int(row["price"])
    user_price = int(item_price*random.uniform(0.75, 0.9))
    return item_name, item_info, item_price, user_price, item_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", dest="item_dataset_path")
    parser.add_argument("-o", "--output_dir", dest="outputdir")
    parser.add_argument("-m", "--max_turn", dest="max_turn")
    args = parser.parse_args()
    gen_conversation(args.item_dataset_path, args.outputdir, args.max_turn)

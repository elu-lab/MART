import openai
import json
import argparse
import tqdm
import time


OPENAI_API_KEY = "YOUR KEY"

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='./G-Eval/prompt/conversation_data/per_prompt.txt')
    argparser.add_argument('--save_fp', type=str, default='./G_Eval/results/8b_distil_result.json')
    argparser.add_argument('--conversation_data_fp', type=str, default='./G_Eval/data/8b_distil_test.json')
    argparser.add_argument('--model', type=str, default='gpt-4-0613')
    args = argparser.parse_args()

    openai.api_key = OPENAI_API_KEY

    conversation_data = json.load(open(args.conversation_data_fp))
    prompt_template = open(args.prompt_fp).read()

    ct, ignore = 0, 0
    new_json = []

    for instance in tqdm.tqdm(conversation_data, desc="Processing instances"):
        chat_history = [] 
        for turn in instance['chat']:
            role = turn.split(":")[0].strip()
            content = turn.split(":")[1].strip()

            if role == "merchant":
                merchant_turn = content
                item_name = instance["environment"]["item_name"]
                item_info = instance["environment"]["item_info"]

                cur_prompt = prompt_template.replace('{{Document}}', "\n".join(chat_history)) \
                                            .replace('{{Response}}', merchant_turn) \
                                            .replace('{{Name}}', item_name) \
                                            .replace('{{Description}}', item_info)

                chat_history.append(f"merchant: {merchant_turn}")

                instance_data = {
                    "environment": instance["environment"],
                    "merchant_turn": merchant_turn,
                    "chat_history": chat_history.copy(),
                    "prompt": cur_prompt
                }

                while True:
                    try:
                        _response = openai.ChatCompletion.create(
                            model=args.model,
                            messages=[{"role": "system", "content": cur_prompt}],
                            temperature=1,
                            max_tokens=5,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None,
                            n=20
                        )
                        time.sleep(0.5)

                        all_responses = [_response['choices'][i]['message']['content'] for i in range(len(_response['choices']))]
                        instance_data['all_responses'] = all_responses
                        new_json.append(instance_data)
                        ct += 1

                        if ct % 100 == 0: 
                            with open(args.save_fp, 'w') as f:
                                json.dump(new_json, f, ensure_ascii=False, indent=4)
                            print(f"Progress saved at {ct} instances.")

                        break
                    except Exception as e:
                        print(e)
                        if "limit" in str(e):
                            time.sleep(2)
                        else:
                            ignore += 1
                            print('ignored', ignore)
                            break

            else:
                chat_history.append(turn)

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)

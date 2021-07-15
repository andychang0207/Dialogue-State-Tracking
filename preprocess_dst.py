import json
import jsonlines
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
#from pprint import pprint
"""
preprocess for DST dataset
"""

# 一般的前處理
def main(args):
    with open(args.schema_path) as f:
        schema = json.load(f)
    dialogues_list = list(args.data_path.glob('*.json'))
    dialogues_list.sort()
    #pprint(dialogues_list)
    processed_data = []
    print("*****-----*****----- preprocessing -----*****-----*****")
    progress_bar = tqdm(total=len(dialogues_list))
    processed_data = []
    for dialogues_path in dialogues_list:
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            history = ""
            belief = ""
            for idx,turn in enumerate(dialogue['turns']):
                # USER turn: collect state and record user utterance
                if turn['speaker'] == "USER":
                    history += " <|user|> "+ turn['utterance'].strip()
                    state = []
                    occur_state = []
                    belief = ""
                    if len(turn['frames']) == 0:
                        continue
                    domain = turn['frames'][0]['service'] #.split("_").lower()
                        
                    for k, v in turn['frames'][0]['state']['slot_values'].items():
                        state.append([domain, k, v])
                        occur_state.append(k)
                    state.sort(key = lambda x: x[0] + " " + x[1])
                    for s in state:
                        s[2].sort()
                        s[2] = s[2][0]
                    if args.add_none_slot:
                        for s in schema:
                            if s['service_name'] == domain:
                                for slot in s['slots']:
                                    if slot['name'] not in occur_state:
                                        state.append([domain,slot['name'],"not mentioned"])
                        state.sort(key = lambda x: x[0] + " " + x[1])
                    state = [s[0] + " " + s[1] + " " + s[2] for s in state]
                    belief = "<|belief|> " + ", ".join(state) + " <|endofbelief|> " if state else "<|belief|> <|endofbelief|> "
                # SYSTEM turn: add in dataset and record system utterance
                else:
                    if len(turn['frames']) == 0:
                        break
                    utterance = turn["utterance"].strip()
                    seq = "<|context|>" + history + " <|endofcontext|> " + belief
                    processed_data.append({"text":"<|endoftext|>"+seq+"<|endoftext|>"})
                    
                    # *****----- record history ------*****
                    history += " <|system|> " + utterance
                    # *****----- record history ------*****
        progress_bar.update(1)
    
    
    output_name = os.path.join(args.output_dir,args.data_name+".jsonl")
    output_json = os.path.join(args.output_dir,args.data_name+".json")
    # 存 jsonl
    with jsonlines.open(output_name,mode="w") as writer:
        for datapoint in processed_data:
            writer.write(datapoint)
    # 存 json
    json.dump(processed_data,open(output_json,'w'),indent=4)
    print("\n")
    print("*****-----*****----- Completed preprocess -----*****-----*****")
    return

# 紀錄所有歷史的states並且當作input
def sequential(args):
    with open(args.schema_path) as f:
        schema = json.load(f)
    dialogues_list = list(args.data_path.glob('*.json'))
    dialogues_list.sort()
    #pprint(dialogues_list)
    processed_data = []
    print("*****-----*****----- preprocessing -----*****-----*****")
    progress_bar = tqdm(total=len(dialogues_list))
    processed_data = []
    for dialogues_path in dialogues_list:
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            history = ""
            belief = ""
            history_state = {}
            history_belief = ""
            for idx,turn in enumerate(dialogue['turns']):
                # USER turn: goal collect state
                if turn['speaker'] == "USER":
                    history += " <|user|> "+ turn['utterance'].strip()
                    state = []
                    occur_state = []
                    belief = ""
                    if len(turn['frames']) == 0:
                        continue
                    domain = turn['frames'][0]['service'] #.split("_").lower()
                        
                    for k, v in turn['frames'][0]['state']['slot_values'].items():
                        state.append([domain, k, v])
                        occur_state.append(k)
                    state.sort(key = lambda x: x[0] + " " + x[1])
                    for s in state:
                        s[2].sort()
                        s[2] = s[2][0]
                    if args.add_none_slot:
                        for s in schema:
                            if s['service_name'] == domain:
                                for slot in s['slots']:
                                    if slot['name'] not in occur_state:
                                        state.append([domain,slot['name'],"not mentioned"])
                        state.sort(key = lambda x: x[0] + " " + x[1])
                    
                    history_state_list = [k+" "+v for k, v in history_state.items()]
                    history_belief = "<|history|> " + ", ".join(history_state_list) + " <|endofhistory|> " if history_state_list else "<|history|> <|endofhistory|> "
                    for s in state:    
                        history_state[s[0]+" "+s[1]] = s[2]
                        
                    state = [s[0] + " " + s[1] + " " + s[2] for s in state]
                    belief = "<|belief|> " + ", ".join(state) + " <|endofbelief|> " if state else "<|belief|> <|endofbelief|> "
                
                else:
                    if len(turn['frames']) == 0:
                        break
                    utterance = turn["utterance"].strip()
                    seq = "<|context|>" + history + " <|endofcontext|> " + history_belief + belief
                    processed_data.append({"text":"<|endoftext|>"+seq+"<|endoftext|>"})
                    
                    # *****----- record history ------*****
                    history += " <|system|> " + utterance
                    # *****----- record history ------*****
                    
        progress_bar.update(1)
    
    
    output_name = os.path.join(args.output_dir,args.data_name+".jsonl")
    output_json = os.path.join(args.output_dir,args.data_name+".json")
    with jsonlines.open(output_name,mode="w") as writer:
        for datapoint in processed_data:
            writer.write(datapoint)
    json.dump(processed_data,open(output_json,'w'),indent=4)
    print("\n")
    print("*****-----*****----- Completed preprocess -----*****-----*****")
    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # *****-----*****----- arguments -----*****-----*****
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./cache/"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default="./data-0625/train"
    )
    parser.add_argument(
        "--schema_path",
        type=Path,
        default="./data/data/schema.json"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="train_dst"
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
    )
    parser.add_argument(
        "--do_sequential",
        action="store_true",
    )
    parser.add_argument(
        "--add_none_slot",
        action="store_true",
    )

    # *****-----*****----- arguments -----*****-----*****
    args = parser.parse_args()
    return args

# test dataset的前處理，只需紀錄對話，schema:
# [
#     {'text':dialogue0 turn0對話  },
#     {'text':dialogue0 turn0-turn1對話  },
#     {'text':dialogue0 turn0-turn2對話  },
#     ...,
#     {'text':dialogueN turn0-turnM對話  }
# ]
# do_sequential 的紀錄方式有些不同，schema:
# [
#   {
#     'dialogue':[
#         {'text':turn0對話},
#         {'text':turn0-turn1對話},
#         ...,
#         {'text':turn0-turnN對話}
#     ]
#   },
# ]
def test_preprocess(args):
    dialogues_list = list(args.data_path.glob('*.json'))
    dialogues_list.sort()
    #pprint(dialogues_list)
    processed_data = []
    processed_data_seq = []
    print("*****-----*****----- preprocessing -----*****-----*****")
    progress_bar = tqdm(total=len(dialogues_list))
    processed_data = []
    for dialogues_path in dialogues_list:
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            processed_dial = []
            history = ""
            for idx,turn in enumerate(dialogue['turns']):
                speaker = " <|user|> " if turn['speaker']=="USER" else " <|system|> "
                history += speaker + turn['utterance'].strip()
                if speaker == " <|user|> ":
                    processed_data.append({'text':"<|context|>"+history+" <|endofcontext|>"})
                    processed_dial.append({'text':"<|context|>"+history+" <|endofcontext|>"})
            processed_data_seq.append({'dialogue':processed_dial})
        progress_bar.update(1)
    output_name = os.path.join(args.output_dir,args.data_name+".jsonl")
    output_json = os.path.join(args.output_dir,args.data_name+".json")
    if args.do_sequential:
        with jsonlines.open(output_name,mode="w") as writer:
            for datapoint in processed_data_seq:
                writer.write(datapoint)
        json.dump(processed_data_seq,open(output_json,'w'),indent=4)
    else:
        with jsonlines.open(output_name,mode="w") as writer:
            for datapoint in processed_data:
                writer.write(datapoint)
        json.dump(processed_data,open(output_json,'w'),indent=4)
    print("\n")
    print("*****-----*****----- Completed preprocess -----*****-----*****")
    return

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.do_test:
        test_preprocess(args)
    elif args.do_sequential:
        sequential(args)
    else:
        main(args)
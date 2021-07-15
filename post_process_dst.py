
import json
import jsonlines
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
from state_to_csv import write_csv
import pickle
#from pprint import pprint
"""
postprocess for chit-chat dataset
"""
def parse_args() -> Namespace:
    parser = ArgumentParser()
    # *****-----*****----- arguments -----*****-----*****
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./test_seen_2_state.json"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default="./prediction/test_seen_2.json"
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default="./data/data/test_seen/"
    )
    parser.add_argument(
        "--none_slot",
        type=bool,
        default=False
    )

    # *****-----*****----- arguments -----*****-----*****
    args = parser.parse_args()
    return args

def main(args):
    error_count = 0
    is_error = False
    with open(args.data_path) as f:
        pred_data = json.load(f)
    output = []
    for pred in pred_data:
        text = pred['text']
        if text.find("<|belief|>") != -1 and text.find("<|endofbelief|>") != -1 and text.count("<|belief|>") == 1 and text.count("<|endofbelief|>") == 1:
            belief = text[text.find("<|belief|>"):text.find("<|endofbelief|>")]
            belief = belief.replace("<|belief|>","").split(',')
            belief = [x.strip() for x in belief]
        else:
            is_error = True
            belief = []
        new_belief = []
        for state in belief:
            s = state.split()
            if len(s) < 3:
                is_error = True
                new_belief.append(state)
            else:
                new_belief.append(state)
        
        output.append({
            'belief':new_belief
        })
        if is_error:
            error_count += 1
            is_error = False
    print("Failure probrability: ",error_count/len(pred_data))
    
    i = 0
    dialogues_list = list(args.test_dir.glob('*.json'))
    dialogues_list.sort()
    processed_data = {}
    print("*****-----*****----- post processing state -----*****-----*****")
    for dialogues_path in dialogues_list:
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            processed_data[dialogue['dialogue_id']] = {}
            for idx,turn in enumerate(dialogue['turns']):
                if turn['speaker'] == "SYSTEM":
                    assert output
                    output_data = output.pop(0)
                    for state in output_data['belief']:
                        state = state.replace("\n"," ").split(" ")
                        if len(state) < 2:
                            continue
                        elif len(state) == 2:
                            domain = state[0].split("-")[0] if len(state[0].split("-")) == 2 else state[0].split("_")[0]
                            if args.none_slot:
                                if (" ".join(state[1:])).strip() != "not mentioned":
                                    processed_data[dialogue['dialogue_id']][f'{domain}-{state[0]}'] = (" ".join(state[1:])).strip()
                            else:
                                processed_data[dialogue['dialogue_id']][f'{domain}-{state[0]}'] = (" ".join(state[1:])).strip()
                            # if i not in failure:
                            #     failure.append(i)
                            # continue
                        else:
                            if args.none_slot:
                                if (" ".join(state[2:])).strip() != "not mentioned":
                                    processed_data[dialogue['dialogue_id']][f'{state[0]}-{state[1]}'] = (" ".join(state[2:])).strip()
                            else:
                                processed_data[dialogue['dialogue_id']][f'{state[0]}-{state[1]}'] = (" ".join(state[2:])).strip()
                        i += 1
    assert not output
    json.dump(processed_data,open(args.output_path,"w"),indent=2)
    write_csv(processed_data,str(args.output_path).replace('json','csv'))
    # if failure:
    #     pickle.dump(failure,open('failure_index.pkl','wb'))

    return



if __name__ == "__main__":
    args = parse_args()
    main(args)
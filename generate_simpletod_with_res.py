#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

from os import truncate
import time
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.functional import softmax

from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
from transformers.models.auto.configuration_auto import replace_list_option_in_docstrings


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--output_path", type=Path, default="./prediction/test1.json")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--beam_search", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--do_generate_all", action="store_true")

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    if args.test_file is None:
        prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        # output_sequences = model.generate(
        #     input_ids=input_ids,
        #     max_length=args.length + len(encoded_prompt[0]),
        #     temperature=args.temperature,
        #     top_k=args.k,
        #     num_beams=args.beam_search,
        #     top_p=args.p,
        #     repetition_penalty=args.repetition_penalty,
        #     do_sample=True,
        #     num_return_sequences=args.num_return_sequences,
        # )
        if not args.do_generate_all:
            response = input("Model response >>> ")
            response_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt")
            response_ids = response_ids.to(args.device)
        break_tokens = tokenizer.encode(tokenizer._eos_token)
        break_to_replace_res = tokenizer.encode("<|endofaction|>")

        model.eval()
        with torch.no_grad():
            predicted_index = input_ids[0][-1]
            while predicted_index not in break_tokens:
                outputs = model(input_ids)
                predictions = outputs[0]
                if args.do_sample:
                        logits = predictions[0, -1, :] / args.temperature
                        filtered_logits = top_k_top_p_filtering(logits,top_k=args.k,top_p=args.p)
                        probabilities = softmax(filtered_logits, dim=-1)
                        next_token = torch.multinomial(probabilities, 1)
                else:
                    next_token = torch.argmax(predictions[0, -1, :])
                predicted_index = next_token.item()
                next_token = next_token.view(1,1)
                input_ids = torch.cat((input_ids,next_token),dim=1)
                if predicted_index in break_to_replace_res and not args.do_generate_all:
                    input_ids = torch.cat((input_ids,response_ids),dim=-1)
                    predicted_index = input_ids[0][-1]
                if input_ids.size()[-1] >= 1024:
                    break

        text = tokenizer.decode(input_ids[0])
        print(text)

        


    else:
        start_time = time.time()
        
        prompt_max_length = 1024 - args.length
        # def preprocess_function(examples):
        #     texts = examples['text']
        #     responses = examples['response']
        #     # texts = [text.split("<|endofcontext|>")[0]+"<|endofcontext|>" for text in raw_texts]
        #     tokenized_texts = tokenizer(texts)
        #     tokenized_responses = tokenizer(responses)
        #     # for i in range(len(tokenized_texts['input_ids'])):
        #     #     if len(tokenized_texts['input_ids'][i]) > prompt_max_length:
        #     #         tokenized_texts['input_ids'][i] = tokenized_texts['input_ids'][i][len(tokenized_texts['input_ids'][i])-prompt_max_length:]
        #     #         tokenized_texts['attention_mask'][i] = tokenized_texts['attention_mask'][i][len(tokenized_texts['attention_mask'][i])-prompt_max_length:]
        #     result = {
        #         "input_ids":tokenized_texts['input_ids'],
        #         "response":tokenized_responses['input_ids']
        #     }
        #     return result
        
        # def collate_fn(features):
        #     first = features[0]
        #     batch = {}
        #     for k, v in first.items():
        #         if v is not None and not isinstance(v, str):
        #             if isinstance(v, torch.Tensor):
        #                 batch[k] = torch.stack([f[k] for f in features])
        #             else:
        #                 batch[k] = torch.tensor([f[k] for f in features])

        #     return batch

        # raw_dataset = load_dataset("json",data_files={'test':str(args.test_file)})
        # processed_dataset = raw_dataset.map(
        #     preprocess_function,
        #     batched=True,
        #     load_from_cache_file=True,
        # )
        with open(args.test_file) as f:
            test_dataset = json.load(f)
        # test_dataset = processed_dataset['test']
        
        # test_dataloader = DataLoader(
        #     test_dataset,
        #     collate_fn=collate_fn,
        #     batch_size=args.per_device_test_batch_size
        # )

        break_tokens = tokenizer.encode(tokenizer._eos_token)
        break_to_replace_res = tokenizer.encode("<|endofaction|>")
        output = []
        progress_bar = tqdm(total = len(test_dataset))
        model.eval()
        for batch in test_dataset:
            encoded_prompt = tokenizer.encode(batch['text'], add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(args.device)
            if not args.do_generate_all:
                response_ids = tokenizer.encode(batch['response'], add_special_tokens=False, return_tensors="pt")
                response_ids = response_ids.to(args.device)
            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt
            
            with torch.no_grad():
                predicted_index = input_ids[0][-1]
                while predicted_index not in break_tokens:
                    outputs = model(input_ids)
                    predictions = outputs[0]
                    if args.do_sample:
                        logits = predictions[0, -1, :] / args.temperature
                        filtered_logits = top_k_top_p_filtering(logits,top_k=args.k,top_p=args.p)
                        probabilities = softmax(filtered_logits, dim=-1)
                        next_token = torch.multinomial(probabilities, 1)
                    else:
                        next_token = torch.argmax(predictions[0, -1, :])
                    predicted_index = next_token.item()
                    next_token = next_token.view(1,1)
                    input_ids = torch.cat((input_ids,next_token),dim=1)
                    if not args.do_generate_all and predicted_index in break_to_replace_res:
                        input_ids = torch.cat((input_ids,response_ids),dim=-1)
                        predicted_index = input_ids[0][-1]
                    if input_ids.size()[-1] >= 1024:
                        # input_ids = input_ids[...,-1024:]
                        break

                

            
            for generated_sequence in input_ids:
                    
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            
                output.append({
                    'text':text
                })
            progress_bar.update(1)

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        
        end_time = time.time()
        print(f"Model execution time: {end_time - start_time} s")
        progress_bar.close()

        if args.output_path is not None:
            json.dump(output,open(args.output_path,"w"),indent=4)



if __name__ == "__main__":
    main()

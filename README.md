
# DST

Dialogue State Tracking based on [simpleTOD](https://arxiv.org/abs/2005.00796) and [simpleTOD+](https://arxiv.org/abs/2010.12757)

and code based on [Huggingface](https://huggingface.co/)
## Installation 

```bash 
pip install -r requirements.txt
```
|package|version|
|---|-----|
|`torch`|1.8.0|
|`datasets`|1.8.0|
|`transformers`|4.5.0|
|`seqeval`|1.2.2|
|`tqdm`||
|`accelerate`|0.3.0|
|`jsonlines`||
    
## Training

To run traing, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/train/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name train_dst
```

```bash
python preprocess_dst.py \
    --data_path {path/to/eval/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name eval_dst
```

```bash
python run_simpletod_plus.py \
    --output_dir ./DST_5epoch_1e-3 \
    --model_name_or_path=gpt2 \
    --model_type=gpt2 \
    --train_file=./cache/train_dst.jsonl \
    --validation_file=./cache/eval_dst.jsonl \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --learning_rate 1e-3
```

To run traing including none state, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/train/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name train_dst_none_slot \
    --add_none_state
```

```bash
python preprocess_dst.py \
    --data_path {path/to/eval/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name eval_dst_none_slot \
    --add_none_state
```

```bash
python run_simpletod_plus.py \
    --output_dir ./DST_with_none_5epoch_1e-3 \
    --model_name_or_path=gpt2 \
    --model_type=gpt2 \
    --train_file=./cache/train_dst_none_slot.jsonl \
    --validation_file=./cache/eval_dst_none_slot.jsonl \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --learning_rate 1e-3
```

To run traing including none state and not concatenate datapoint, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/train/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name train_dst_none_slot \
    --add_none_state
```

```bash
python preprocess_dst.py \
    --data_path {path/to/eval/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name eval_dst_none_slot \
    --add_none_state
```

```bash
python run_simpletod_plus_seperate.py \
    --output_dir ./DST_with_none_5epoch_seperate_5e-5 \
    --model_name_or_path=gpt2 \
    --model_type=gpt2 \
    --train_file=./cache/train_dst_none_slot.jsonl \
    --validation_file=./cache/eval_dst_none_slot.jsonl \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 7 \
    --learning_rate 5e-5
```

example of DST training data including none slot

```
<|endoftext|><|context|> <|user|> I need some bus tickets. <|system|> Sure, i can help you. Where are you departing from? <|user|> San Diego on the 8th of this month. <|endofcontext|>
<|belief|> Buses_2 departure_date 8th of this month, Buses_2 departure_time not mentioned, Buses_2 destination not mentioned, Buses_2 destination_station_name not mentioned, Buses_2 fare_type not mentioned, Buses_2 group_size not mentioned, Buses_2 origin San Diego, Buses_2 origin_station_name not mentioned, Buses_2 price not mentioned <|endofbelief|> <|endoftext|>
```

To run traing including history belief, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/train/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name train_dst_seeq \
    --do_sequential
```

```bash
python preprocess_dst.py \
    --data_path {path/to/eval/data/folder} \
    --schema_path {path/to/schema.json} \
    --data_name eval_dst_seq \
    --do_sequential
```

```bash
python run_simpletod_plus.py \
    --output_dir ./DST_with_history_belief_5epoch_1e-3 \
    --model_name_or_path=gpt2 \
    --model_type=gpt2 \
    --train_file=./cache/train_dst_seq.jsonl \
    --validation_file=./cache/eval_dst_seq.jsonl \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --learning_rate 1e-3
```

example of DST training data including history beleif

```
<|endoftext|><|context|> <|user|> I need some bus tickets. <|system|> Sure, i can help you. Where are you departing from? <|user|> San Diego on the 8th of this month. <|system|> ok, how many tickets and where are you traveling too? What time would you like to depart? <|user|> I need one ticket to Anaheim and if I could leave at 4:15 pm would be great. <|endofcontext|> 
<|history|> Buses_2 departure_date 8th of this month, Buses_2 origin San Diego <|endofhistory|> 
<|belief|> Buses_2 departure_date 8th of this month, Buses_2 departure_time 4:15 pm, Buses_2 destination Anaheim, Buses_2 group_size 1, Buses_2 origin San Diego <|endofbelief|> <|endoftext|>
```

## Generation

To run DST, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/test/data/folder} \
    --data_name test_dst \
    --do_test
```

```bash
python generate_simpletod_with_res.py \
    --model_type=gpt2 \
    --model_name_or_path ./DST_5epoch_1e-3 \
    --test_file=./cache/test_dst.json \
    --output_path {path/to/prediction.json} \
    --do_generate_all
```

```bash
python post_process_dst.py \
    --output_path {path/to/output/processed/prediction.json} \
    --data_path {path/to/prediction.json} \
    --test_dir {path/to/test/data/folder} 
```

To run DST including none state, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/test/data/folder} \
    --data_name test_dst \
    --do_test
```

```bash
python generate_simpletod_with_res.py \
    --model_type=gpt2 \
    --model_name_or_path ./DST_with_none_5epoch_1e-3 \
    --test_file=./cache/test_dst.json \
    --output_path {path/to/prediction.json} \
    --do_generate_all
```

```bash
python post_process_dst.py \
    --output_path {path/to/output/processed/prediction.json} \
    --data_path {path/to/prediction.json} \
    --test_dir {path/to/test/data/folder} \
    --none_slot True
```

To run DST including none state and run in batch, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/test/data/folder} \
    --data_name test_dst \
    --do_test
```

```bash
python generate_simpletod.py \
    --model_type=gpt2 \
    --model_name_or_path ./DST_with_none_5epoch_1e-3 \
    --test_file=./cache/test_dst.json \
    --output_path {path/to/prediction.json} \
    --do_batch \
    --beam_search 5 \
    --do_sample False \
    --per_device_test_batch_size 8 \
    --stop_token "<|endoftext|>"
```

```bash
python post_process_dst.py \
    --output_path {path/to/output/processed/prediction.json} \
    --data_path {path/to/prediction.json} \
    --test_dir {path/to/test/data/folder} \
    --none_slot True
```

To run DST including history belief, run the following command

```bash
python preprocess_dst.py \
    --data_path {path/to/test/data/folder} \
    --data_name test_dst \
    --do_sequential \
    --do_test
```

```bash
python generate_dst_with_history_belief.py \
    --model_type=gpt2 \
    --model_name_or_path ./DST_with_history_beleif_5epoch_1e-3 \
    --test_file=./cache/test_dst.json \
    --output_path {path/to/prediction.json}
```

```bash
python post_process_dst.py \
    --output_path {path/to/output/processed/prediction.json} \
    --data_path {path/to/prediction.json} \
    --test_dir {path/to/test/data/folder} 
```



## Parameter Reference

### run_simpletod_plus.py

fine-tuning gpt2 model

| Parameter | Default     | Description                |
| :-------- | :------- | :------------------------- |
|`-h`, `--help`| |show the help message and exit|
|`--dataset_name DATASET_NAME`| |The name of the dataset to use (via the datasets library).|
|`--dataset_config_name DATASET_CONFIG_NAME`| |The configuration name of the dataset to use (via the datasets library).|
|`--train_file TRAIN_FILE`| |A csv or a jsonl file containing the training data.|
|`--validation_file VALIDATION_FILE`| |A csv or a jsonl file containing the validation data.|
|`--model_name_or_path MODEL_NAME_OR_PATH`| |**Recommend gpt2**, Path to pretrained model or model identifier from huggingface.co/models.|
|`--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE`| 8|Batch size (per device) for the training dataloader.|
|`--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE`|8|Batch size (per device) for the evaluation dataloader.|
|`--learning_rate LEARNING_RATE`|5e-5|Initial learning rate (after the potential warmup period) to use.|
|`--weight_decay WEIGHT_DECAY`|0.0|Weight decay to use.|
|`--num_train_epochs NUM_TRAIN_EPOCHS`|3|Total number of training epochs to perform.|
|`--max_train_steps MAX_TRAIN_STEPS`| |Total number of training steps to perform. If provided, overrides num_train_epochs.|
|`--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS`|1|Number of updates steps to accumulate before performing a backward/update pass.|
|`--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}`|`linear`|The scheduler type to use.|
|`--num_warmup_steps NUM_WARMUP_STEPS`|0|Number of steps for the warmup in the lr scheduler.|
|`--output_dir OUTPUT_DIR`| |Where to store the final model.|
|`--seed SEED`|42 |A seed for reproducible training.|
|`--block_size BLOCK_SIZE`| |Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).|
|`--preprocessing_num_workers PREPROCESSING_NUM_WORKERS`|4|The number of processes to use for the preprocessing.|
|`--overwrite_cache OVERWRITE_CACHE`|`False`|Overwrite the cached training and evaluation sets|

### preprocess_dst.py

preprocessing with dst training dataset, eval dataset or test dataset

| Parameter | Default     | Description                |
| :-------- | :------- | :------------------------- |
|`-h`, `--help`|      |show the help message and exit|
|`--output_dir OUTPUT_DIR`|`./cache/` |Where to store the preprocessed data.|
|`--data_path DATA_PATH`| `./data-0625/train`|A directory containing the training data.|
|`--schema_path SCHEMA_PATH`| `./data/data/schema.json`|A json file containing the schema.|
|`--data_name DATA_NAME`| `train_dst`|A name of preprocessed data file.|
|`--do_test`| |If passed, it will preprocess data to test dataset|
|`--do_sequential`| | If passed, it will preprocess data to dataset of DST including history states|
|`--add_none_slot`| |If passed, it will preprocess data to dataset of DST including none state|

### post_process_dst.py

post processing prediction data to the format that kaggle required.

| Parameter | Default     | Description                |
| :-------- | :------- | :------------------------- |
|`-h`, `--help `|  | show the help message and exit|
|`--output_path OUTPUT_PATH`|`./test_seen_2_state.json`|Where to store the post-processed data.|
|`--data_path DATA_PATH`|`./prediction/test_seen_2.json`|A path to prediction json file|
|`--test_dir TEST_DIR`|`./data/data/test_seen/`|A directory containing the test dataset|
|`--none_slot NONE_SLOT`|`False`|Whether ignore the none state or not.|

### generate_simpletod_with_res.py

generate the DST prediction

| Parameter | Default     | Description                |
| :-------- | :------- | :------------------------- |
|`-h`, `--help `|  | show the help message and exit|
|`--model_type MODEL_TYPE`| |**Must be same as the pretrained model**, Model type selected in the list: gpt2, ctrl, openai-gpt, xlnet, transfo-xl, xlm|
|`--model_name_or_path MODEL_NAME_OR_PATH`| |Path to directory where stores model weight|
|`--output_path OUTPUT_PATH`|`./prediction/test1.json`|Where to store predition|
|`--test_file TEST_FILE`| |Path to test dataset|
|`--temperature TEMPERATURE`|1.0|temperature of 1.0 has no effect, lower tend toward greedy sampling|
|`--repetition_penalty REPETITION_PENALTY`| 1.0|primarily useful for CTRL model; in that case, use 1.2|
|`--k K`| 0|top k sampling|
|`--do_sample`| |If passed, use sampling decoding algorithm.|
|`--p P`| 0.9|nucleus sampling|
|`--do_generate_all`| | If passed, generate to the endoftext, not stop to replace system response|
|`--seed SEED`|42| random seed for initialization|
|`--no_cuda ` | |Avoid using CUDA when available|
|`--fp16`| |Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit|

### generate_dst_with_history_belief.py

generate the DST prediction with history states
| Parameter | Default     | Description                |
| :-------- | :------- | :------------------------- |
|`-h`, `--help `|  | show the help message and exit|
|`--model_type MODEL_TYPE`| |**Must be same as the pretrained model**, Model type selected in the list: gpt2, ctrl, openai-gpt, xlnet, transfo-xl, xlm|
|`--model_name_or_path MODEL_NAME_OR_PATH`| |Path to directory where stores model weight|
|`--output_path OUTPUT_PATH`|`./prediction/test1.json`|Where to store predition|
|`--test_file TEST_FILE`| |Path to test dataset|
|`--temperature TEMPERATURE`|1.0|temperature of 1.0 has no effect, lower tend toward greedy sampling|
|`--repetition_penalty REPETITION_PENALTY`| 1.0|primarily useful for CTRL model; in that case, use 1.2|
|`--k K`| 0|top k sampling|
|`--do_sample`| |If passed, use sampling decoding algorithm.|
|`--p P`| 0.9|nucleus sampling|
|`--do_generate_all`| | If passed, generate to the endoftext, not stop to replace system response|
|`--seed SEED`|42| random seed for initialization|
|`--no_cuda ` | |Avoid using CUDA when available|
|`--fp16`| |Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit|

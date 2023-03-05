from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, set_seed

from relation_extraction_task import get_prepared_dataset as get_re_dataset
from nli_task import get_prepared_dataset as get_nli_dataset

import datasets

# Import the W&B Python Library 
import os

os.environ['WANDB_PROJECT']   = 'glm_re'
os.environ['WANDB_LOG_MODEL'] = 'false'
os.environ['WANDB_WATCH']     = 'false'

set_seed(1)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model     = AutoModelForCausalLM.from_pretrained("gpt2")
dataset_re   = get_re_dataset(tokenizer=tokenizer)
dataset_nli  = get_nli_dataset(tokenizer=tokenizer)
dataset = datasets.interleave_datasets([dataset_re['train'].filter(lambda x: len(x['input_ids']) < 112), dataset_nli['train'].filter(lambda x: len(x['input_ids']) < 112)])

print(dataset)

# collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=512, padding=True)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)


output_dir = "output3"

training_args = Seq2SeqTrainingArguments(
    output_dir                  = output_dir,
    fp16                        = True,
    # fp16_backend                = "amp",
    per_device_train_batch_size = 64,
    per_device_eval_batch_size  = 64,
    # eval_accumulation_steps     = 16,
    # evaluation_strategy         = "steps",
    # eval_steps                  = 5000,      #logging_steps,
    save_steps                  = 10000,
    logging_steps               = 50,
    save_total_limit            = 6,
    max_steps                   = 50000,
    gradient_accumulation_steps = 2,
    report_to                   = "wandb",
    remove_unused_columns       = False,
    # weight_decay                = 0.001,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = 'linear',
    dataloader_num_workers      = 16,
    learning_rate               = 3e-4,
)


trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = dataset,
    # eval_dataset    = dataset['validation'],
    # eval_dataset    = {'en_ner': en_ner['validation'].select(range(1000)), 'fr_ner': fr_ner['validation'].select(range(1000))},
    tokenizer       = tokenizer,
    data_collator   = collator,
    # compute_metrics = {'en_ner': lambda x: compute_metrics(x[0], x[1], 'en_ner'), 'fr_ner': lambda x: compute_metrics(x[0], x[1], 'fr_ner'), }
)

trainer.train()


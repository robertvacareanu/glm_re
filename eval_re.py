import torch
import tqdm
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, set_seed

from relation_extraction_task import get_prepared_dataset

# Import the W&B Python Library 
import os

os.environ['WANDB_PROJECT']   = 'glm_re'
os.environ['WANDB_LOG_MODEL'] = 'false'
os.environ['WANDB_WATCH']     = 'false'

set_seed(1)

device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained('/home/rvacareanu/projects_6_2301/glm_re/output/checkpoint-50000')
tokenizer.pad_token = tokenizer.eos_token
model     = AutoModelForCausalLM.from_pretrained('/home/rvacareanu/projects_6_2301/glm_re/output/checkpoint-50000').to(device)
dataset   = get_prepared_dataset(tokenizer=tokenizer)

# collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=512, padding=True)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(tokenizer.decode(dataset['validation'][0]['input_ids']))
# print(tokenizer.decode(dataset['validation'][0]['input_ids'][-3:]))
print(dataset['validation'])
correct = 0
total  = 0
output = []
for line in tqdm.tqdm(dataset['validation']):
    ii = line['input_ids'][:-3]
    eo = line['input_ids'][-3:]
    generated = model.generate(input_ids=torch.tensor(ii).reshape(1, -1).to(device), max_length=len(line['input_ids']))[0][-3:].detach().cpu().tolist()
    output.append((line['id'], line['relation_test'], line['relation_gold'], generated))

from collections import defaultdict

output = [(x[0], x[1], x[2], tokenizer.decode(x[3]).strip()) for x in output]

output_dd = defaultdict(list)

for x in output:
    output_dd[x[0]].append(x)


import random
random.seed(1)

gold = []
pred = []
for key in output_dd.keys():
    all_preds = output_dd[key]
    gold.append(all_preds[0][2])
    all_preds = [(x[1], x[3]) for x in all_preds if x[3] == 'Yes']
    if len(all_preds) == 0:
        pred.append('no_relation')
    else:
        pred.append(random.choice(all_preds)[0])

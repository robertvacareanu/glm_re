import torch
import tqdm
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, set_seed

from relation_extraction_task import get_prepared_dataset, tacred_score

# Import the W&B Python Library 
import os

os.environ['WANDB_PROJECT']   = 'glm_re'
os.environ['WANDB_LOG_MODEL'] = 'false'
os.environ['WANDB_WATCH']     = 'false'

set_seed(1)

device = torch.device('cuda:1')

tokenizer = AutoTokenizer.from_pretrained('/home/rvacareanu/projects_6_2301/glm_re/output2/checkpoint-20000')
model     = AutoModelForCausalLM.from_pretrained('/home/rvacareanu/projects_6_2301/glm_re/output2/checkpoint-20000').to(device).half()

tokenizer.padding_side = "left"

# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

dataset   = get_prepared_dataset(tokenizer=tokenizer, with_answer=False)


# collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=512, padding=True)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print(tokenizer.decode(dataset['validation'][0]['input_ids']))
# print(tokenizer.decode(dataset['validation'][0]['input_ids'][-3:]))
dl = torch.utils.data.DataLoader(dataset['validation'].remove_columns(['id', 'relation_test', 'relation_gold']), batch_size=32, shuffle=False, collate_fn=collator, num_workers=32)


print(dataset['validation'])
correct = 0
total  = 0
output = []
for batch in tqdm.tqdm(dl):
    generated = model.generate(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), max_length=len(batch['input_ids'][0]) + 3).detach().cpu().tolist()
    decoded = [tokenizer.decode(x[len(batch['input_ids'][0]):][2]).strip() for x in generated]
    output += decoded





from collections import defaultdict

output = [(x['id'], x['relation_test'], x['relation_gold'], y) for (x, y) in zip(dataset['validation'], output)]

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
        # pred.append(output_dd[key][0][2])

tacred_score(gold, pred, verbose=True)

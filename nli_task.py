from datasets import load_dataset, DatasetDict, Dataset

from typing import List, Dict

def get_dataset():
    dataset = load_dataset('snli')
    labels  = dataset['train'].features['label'].names

    collapsed_label = ['Yes', 'No', 'No', ]

    data_train = []
    data_val   = []
    data_test  = []
    for split, destination in zip(['train', 'validation', 'test'], [data_train, data_val, data_test]):
        for line in dataset[split]:
            destination.append({**line, 'collapsed_label': collapsed_label[line['label']]})
    
    return DatasetDict({
        'train'     : Dataset.from_list(data_train),
        'validation': Dataset.from_list(data_val),
        'test'      : Dataset.from_list(data_test),
    })



def get_prepared_dataset(tokenizer, **args):
    d = get_dataset()


    max_length   = args.get('max_length', 512)
    with_answer  = args.get('with_answer', True)

    train      = []
    validation = []
    test       = []
    
    # Not exactly necessary
    # We might be interested in adding additional details to the prompts
    # Like the language, the task (although this is already known), or the
    # dataset
    # The idea is that it might be helpful to add these informations
    additional_details = {
        'language': 'en',
        'dataset' : 'snli',
        'task'    : 'nli'
    }

    
    for i, line in enumerate(d['train']):
        template = get_template(line['premise'], hypothesis=line['hypothesis'], answer=line['collapsed_label'], eos_token=tokenizer.eos_token, with_answer=with_answer, **additional_details)
        train.append({**template})
    for i, line in enumerate(d['validation']):
        template = get_template(line['premise'], hypothesis=line['hypothesis'], answer=line['collapsed_label'], eos_token=tokenizer.eos_token, with_answer=with_answer, **additional_details)
        validation.append({**template})
    for i, line in enumerate(d['test']):
        template = get_template(line['premise'], hypothesis=line['hypothesis'], answer=line['collapsed_label'], eos_token=tokenizer.eos_token, with_answer=with_answer, **additional_details)
        test.append({**template})

    return DatasetDict({
        'train'     : Dataset.from_list(train),
        'validation': Dataset.from_list(validation),
        'test'      : Dataset.from_list(test),
    }).map(lambda x: {
        **tokenizer(x['input'], truncation=True, max_length=max_length), 
    }, batched=True, load_from_cache_file=False).remove_columns(['input'])

def get_template(text: str, hypothesis: str, answer: str, language, dataset, task='relation_extraction', eos_token='', with_answer=True) -> List[Dict[str, str]]:
    inp = f'Given the following text:\n########\n{text}\n########\nIs the following true \n########\n{hypothesis}\n########\n'
    if with_answer:
        result = {
            'input' : f'{inp}Answer: {answer} {eos_token}',
        }
    else:
        result = {
            'input' : f'{inp}',#Answer: {answer} {eos_token}',
        }

    return result

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    dataset = get_prepared_dataset(tokenizer=tok)
    print(tok.decode(dataset['train'][0]['input_ids']))

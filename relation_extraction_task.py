"""
This file handles all the necessary code for loading the `TACRED` dataset
For now we use the files from Huggingface. But this works with data on disk too


Language: English
Task    : Relation Extraction
"""

import json

from datasets import load_dataset, DatasetDict, Dataset

from string import Template

from typing import List, Dict

relation_to_text = {
    'no_relation'                        : Template('no relation between $SUBJ and $OBJ'),
    'org:alternate_names'                : Template('$SUBJ is also called $OBJ'),
    'org:city_of_headquarters'           : Template('$SUBJ is located in $OBJ'),
    'org:country_of_headquarters'        : Template('$SUBJ is located in $OBJ'),
    'org:dissolved'                      : Template('$SUBJ dissolved in $OBJ'),
    'org:founded'                        : Template('$SUBJ was founded in $OBJ'),
    'org:founded_by'                     : Template('$SUBJ was founded by $OBJ'),
    'org:member_of'                      : Template('$SUBJ is a member of $OBJ'),
    'org:members'                        : Template('$OBJ is a member of $SUBJ'),
    'org:number_of_employees/members'    : Template('$SUBJ employs $OBJ people'),
    'org:parents'                        : Template('$SUBJ is a subsidiary $OBJ'),
    'org:political/religious_affiliation': Template('$SUBJ has political/religious affiliation with $OBJ'),
    'org:shareholders'                   : Template('$OBJ invested in $SUBJ'),
    'org:stateorprovince_of_headquarters': Template('$SUBJ is located in $OBJ'),
    'org:subsidiaries'                   : Template('$OBJ is a subsidiary of $SUBJ'),
    'org:top_members/employees'          : Template('$OBJ is a top employee of $SUBJ'),
    'org:website'                        : Template('$OBJ is the website of $SUBJ'),
    'per:age'                            : Template('$SUBJ is $OBJ years old'),
    'per:alternate_names'                : Template('$SUBJ also called $OBJ'),
    'per:cause_of_death'                 : Template('$SUBJ died because of $OBJ'),
    'per:charges'                        : Template('$SUBJ is charged with $OBJ'),
    'per:children'                       : Template('$SUBJ is the parent of $OBJ'),
    'per:cities_of_residence'            : Template('$SUBJ lives in $OBJ'),
    'per:city_of_birth'                  : Template('$SUBJ was born in $OBJ'),
    'per:city_of_death'                  : Template('$SUBJ died in $OBJ'),
    'per:countries_of_residence'         : Template('$SUBJ lives in $OBJ'),
    'per:country_of_birth'               : Template('$SUBJ lives in $OBJ'),
    'per:country_of_death'               : Template('$SUBJ died in $OBJ'),
    'per:date_of_birth'                  : Template('$SUBJ was born in $OBJ'),
    'per:date_of_death'                  : Template('$SUBJ died in $OBJ'),
    'per:employee_of'                    : Template('$SUBJ works for $OBJ'),
    'per:origin'                         : Template('$OBJ is the nationality of $SUBJ'),
    'per:other_family'                   : Template('$SUBJ and $OBJ are family'),
    'per:parents'                        : Template('$SUBJ is the parent of $OBJ'),
    'per:religion'                       : Template('$SUBJ is the religion of $OBJ'),
    'per:schools_attended'               : Template('$SUBJ attented $OBJ'),
    'per:siblings'                       : Template('$SUBJ and $OBJ are sibblings'),
    'per:spouse'                         : Template('$SUBJ and $OBJ are married'),
    'per:stateorprovince_of_birth'       : Template('$SUBJ was born in $OBJ'),
    'per:stateorprovince_of_death'       : Template('$SUBJ died in $OBJ'),
    'per:stateorprovinces_of_residence'  : Template('$SUBJ lives in $OBJ'),
    'per:title'                          : Template('$SUBJ is a $OBJ'),
}

# Not all entity pairs accept all relations
# In this dictioanry we map between entity pair to valid relations
entity_types_to_valid_relations = {
    ('ORGANIZATION', 'PERSON')           : ['org:founded_by', 'org:shareholders', 'org:top_members/employees'],
    ('PERSON'      , 'PERSON')           : ['per:alternate_names', 'per:children', 'per:other_family', 'per:parents', 'per:siblings', 'per:spouse'],
    ('ORGANIZATION', 'ORGANIZATION')     : ['org:alternate_names', 'org:member_of', 'org:members', 'org:parents', 'org:shareholders', 'org:subsidiaries'],
    ('ORGANIZATION', 'NUMBER')           : ['org:number_of_employees/members'],
    ('ORGANIZATION', 'DATE')             : ['org:dissolved', 'org:founded'],
    ('PERSON'      , 'ORGANIZATION')     : ['per:employee_of', 'per:schools_attended'],
    ('PERSON'      , 'NATIONALITY')      : ['per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:origin'],
    ('PERSON'      , 'LOCATION')         : ['per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death', 'per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:employee_of', 'per:origin', 'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence'],
    ('PERSON'      , 'TITLE')            : ['per:title'],
    ('PERSON'      , 'DATE')             : ['per:date_of_birth', 'per:date_of_death'],
    ('PERSON'      , 'CITY')             : ['per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death'],
    ('ORGANIZATION', 'MISC')             : ['org:alternate_names'],
    ('PERSON'      , 'COUNTRY')          : ['per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:origin'],
    ('PERSON'      , 'MISC')             : ['per:alternate_names'],
    ('PERSON'      , 'CRIMINAL_CHARGE')  : ['per:charges'],
    ('ORGANIZATION', 'CITY')             : ['org:city_of_headquarters'],
    ('ORGANIZATION', 'LOCATION')         : ['org:city_of_headquarters', 'org:country_of_headquarters', 'org:member_of', 'org:parents', 'org:stateorprovince_of_headquarters', 'org:subsidiaries'],
    ('PERSON'      , 'RELIGION')         : ['per:religion'],
    ('PERSON'      , 'NUMBER')           : ['per:age'],
    ('PERSON'      , 'DURATION')         : ['per:age'],
    ('ORGANIZATION', 'RELIGION')         : ['org:political/religious_affiliation'],
    ('ORGANIZATION', 'URL')              : ['org:website'],
    ('PERSON'      , 'STATE_OR_PROVINCE'): ['per:stateorprovince_of_birth', 'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence'],
    ('ORGANIZATION', 'COUNTRY')          : ['org:country_of_headquarters', 'org:member_of', 'org:members', 'org:parents', 'org:subsidiaries'],
    ('ORGANIZATION', 'STATE_OR_PROVINCE'): ['org:member_of', 'org:parents', 'org:stateorprovince_of_headquarters'],
    ('ORGANIZATION', 'IDEOLOGY')         : ['org:political/religious_affiliation'],
    ('PERSON'      , 'CAUSE_OF_DEATH')   : ['per:cause_of_death']
}

entity_name_to_verbalization = {
    'CAUSE_OF_DEATH'   : 'cause of death',
    'CITY'             : 'city',
    'COUNTRY'          : 'country',
    'CRIMINAL_CHARGE'  : 'criminal charge',
    'DATE'             : 'date',
    'DURATION'         : 'duration',
    'IDEOLOGY'         : 'ideology',
    'LOCATION'         : 'location',
    'MISC'             : 'misc',
    'NATIONALITY'      : 'nationality',
    'NUMBER'           : 'number',
    'ORGANIZATION'     : 'organization',
    'PERSON'           : 'person',
    'RELIGION'         : 'religion',
    'STATE_OR_PROVINCE': 'state or province',
    'TITLE'            : 'title',
    'URL'              : 'url',
}

def get_dataset():
    dataset = load_dataset('DFKI-SLT/tacred', data_dir='/data/nlp/corpora/mlmtl/data/tacred/tacred/data/json')
    relations = dataset['train'].features['relation'].names
    subj_type = dataset['train'].features['subj_type'].names
    obj_type  = dataset['train'].features['obj_type'].names

    data_train = []
    data_val   = []
    data_test  = []
    for split, destination in zip(['train', 'validation', 'test'], [data_train, data_val, data_test]):
        for line in dataset[split]:
            destination.append({**line, 'subject': line['token'][line['subj_start']:line['subj_end']], 'object': line['token'][line['obj_start']:line['obj_end']], 'subj_type': subj_type[line['subj_type']], 'obj_type': obj_type[line['obj_type']], 'relation': relations[line['relation']]})
    
    return DatasetDict({
        'train'     : Dataset.from_list(data_train),
        'validation': Dataset.from_list(data_val),
        'test'      : Dataset.from_list(data_test),
    })



def get_prepared_dataset(tokenizer, **args):
    d = get_dataset()


    max_length   = args.get('max_length', 512)
    with_answer  = args.get('with_answer', True) # Whether to append the answer or not. When not, this means we are interested in generating the answer

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
        'dataset' : 'tacred',
        'task'    : 're'
    }

    
    for i, line in enumerate(d['train']):
        valid_templates = {k:relation_to_text[k] for k in entity_types_to_valid_relations[(line['subj_type'], line['obj_type'])]}
        subj      = ' '.join(line['subject'])
        obj       = ' '.join(line['object'])
        subj_type = entity_name_to_verbalization[line['subj_type']]
        obj_type  = entity_name_to_verbalization[line['obj_type']]
        for example in get_template(' '.join(line['token']), templates=valid_templates, subj=subj, obj=obj, subj_type=subj_type, obj_type=obj_type, relation=line['relation'], eos_token=tokenizer.eos_token, with_answer=with_answer, **additional_details):
            # train.append({**example})
            train.append({**example, 'id': i})
    for i, line in enumerate(d['validation']):
        valid_templates = {k:relation_to_text[k] for k in entity_types_to_valid_relations[(line['subj_type'], line['obj_type'])]}
        subj      = ' '.join(line['subject'])
        obj       = ' '.join(line['object'])
        subj_type = entity_name_to_verbalization[line['subj_type']]
        obj_type  = entity_name_to_verbalization[line['obj_type']]
        for example in get_template(' '.join(line['token']), templates=valid_templates, subj=subj, obj=obj, subj_type=subj_type, obj_type=obj_type, relation=line['relation'], eos_token=tokenizer.eos_token, with_answer=with_answer, **additional_details):
            # validation.append({**example})
            validation.append({**example, 'id': i})
    for i, line in enumerate(d['test']):
        valid_templates = {k:relation_to_text[k] for k in entity_types_to_valid_relations[(line['subj_type'], line['obj_type'])]}
        subj      = ' '.join(line['subject'])
        obj       = ' '.join(line['object'])
        subj_type = entity_name_to_verbalization[line['subj_type']]
        obj_type  = entity_name_to_verbalization[line['obj_type']]
        for example in get_template(' '.join(line['token']), templates=valid_templates, subj=subj, obj=obj, subj_type=subj_type, obj_type=obj_type, relation=line['relation'], eos_token=tokenizer.eos_token, with_answer=with_answer, **additional_details):
            # test.append({**example})
            test.append({**example, 'id': i})

    return DatasetDict({
        'train'     : Dataset.from_list(train),
        'validation': Dataset.from_list(validation),
        'test'      : Dataset.from_list(test),
    }).map(lambda x: {
        **tokenizer(x['input'], truncation=True, max_length=max_length), 
    }, batched=True, load_from_cache_file=False).remove_columns(['input'])

def get_template(text: str, templates: Dict[str, Template], subj, obj, subj_type, obj_type, relation, language, dataset, task='relation_extraction', eos_token='', with_answer=True) -> List[Dict[str, str]]:
    result = []
    for (relation_name, template) in templates.items():
        relation_template = template.substitute(SUBJ=subj, OBJ=obj, SUBJTYPE=subj_type, OBJTYPE=obj_type)
        inp = f'Given the following text:\n########\n{text}\n########\nIs the following true \n########\n{relation_template}\n########\n'
        if relation_name == relation:
            answer = "Yes"
        else:
            answer = "No"
        current_dict = {}
        current_dict['relation_test'] = relation_name
        current_dict['relation_gold'] = relation

        if with_answer:
            current_dict['input'] = f'{inp}Answer: {answer} {eos_token}'   
        else:
            current_dict['input'] = f'{inp}'

        result.append(current_dict)

    return result


from collections import Counter, defaultdict
import math

import sys
from typing import Dict, List

NO_RELATION = "no_relation"

def tacred_score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if verbose:
        print( "Precision (micro): {:.2%}".format(prec_micro) ) 
        print( "   Recall (micro): {:.2%}".format(recall_micro) )
        print( "       F1 (micro): {:.2%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    dataset = get_prepared_dataset(tokenizer=tok)
    print(dataset)
    print(tok.decode(dataset['train'][0]['input_ids']))



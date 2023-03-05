from typing import List

from collections import defaultdict

from datasets import Dataset

def groupby(dataset, column_name: str) -> List:

    group = defaultdict(list)

    for line in dataset:
        group[line[column_name]].append(line)

    return [Dataset.from_list(x) for x in group.values()], list(group.keys())
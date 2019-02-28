import numpy as np
import pandas as pd
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting group')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


df = pd.read_csv('datasets/breast-cancer-wisconsin.data.txt')
df.replace('?', -999999, inplace=True)
df.drop(['id'], 1, inplace=True)



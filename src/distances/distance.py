import warnings
from typing import Tuple, List

import scipy.spatial.distance as ssd


# ha 1,5 - 2,0 m távolságban vannak - risky
# ha 1,5 m-nél közelebb vannak egymástól - critic
# dist - mennyi pixel meterben
def distance_calc(centerp, dist) -> Tuple[List[List], List[List], List[int]]:
    # távolságok

    if len(centerp) == 0:
        warnings.warn('CenterP length is 0, returning')
        return [], [], []
    distances = ssd.squareform(ssd.pdist(centerp, 'euclidean'))

    # kritikus és kockázatos távolságok megkeresése
    # egyenlőre legyen 150 pixel 1 méter
    # dist = 150
    # dist = 1000 mm-re mennyi pixel jut
    risky = []
    critic = []
    safe_idx = []
    not_safe_idx = []

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            if i < j:
                if distances[i][j] < 2 * dist:
                    not_safe_idx.append(j)
                    not_safe_idx.append(i)
                    if distances[i][j] < 1.5 * dist:
                        critic.append([centerp[i][0], centerp[i][1], centerp[j][0], centerp[j][1]])
                    else:
                        risky.append([centerp[i][0], centerp[i][1], centerp[j][0], centerp[j][1]])
        if i not in not_safe_idx:
            safe_idx.append(i)

    return risky, critic, safe_idx

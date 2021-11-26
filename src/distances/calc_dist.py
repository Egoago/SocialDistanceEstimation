import warnings
from typing import Tuple, List

import scipy.spatial.distance as ssd


# calculate the distance from 3d coords, and giving back the risky, critic and safe coords, idxs
# dist = 1000 pixel / meter
def calc_dist(centerp, cps, dist=1000) -> Tuple[List[int], List[int], List[int], List[List], List[List]]:
    # distance between all points
    distances = ssd.squareform(ssd.pdist(centerp, 'euclidean'))

    # risky lines coords, critic lines coords, the idx of the risky people, the idx of the critic people
    risky = []
    critic = []
    risky_idx = []
    critic_idx = []

    safe_idx = []
    not_safe_idx = []

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            if i < j:
                if distances[i][j] < 2 * dist:
                    not_safe_idx.append(j)
                    not_safe_idx.append(i)
                    if distances[i][j] < 1.5 * dist:
                        if i not in critic_idx:
                            critic_idx.append(i)
                        if j not in critic_idx:
                            critic_idx.append(j)
                        critic.append([cps[i][0], cps[i][1], cps[j][0], cps[j][1], centerp[i][0], centerp[i][1],
                                       centerp[j][0], centerp[j][1]])
                    else:
                        if i not in risky_idx:
                            risky_idx.append(i)
                        if j not in risky_idx:
                            risky_idx.append(j)
                        risky.append([cps[i][0], cps[i][1], cps[j][0], cps[j][1], centerp[i][0], centerp[i][1],
                                      centerp[j][0], centerp[j][1]])
        if i not in not_safe_idx:
            safe_idx.append(i)

    return risky_idx, critic_idx, safe_idx, risky, critic

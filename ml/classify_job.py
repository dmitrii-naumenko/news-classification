import time
import numpy as np

def test_job(i):
    print(i)
    time.sleep(i/5)
    print(i)
    return i * 2

def job(i, idxs_text, min_length, threshold):
    A = idxs_text[i]
    if len(A) < min_length:
        return []

    len_A = len(A)
    simularity = np.array([len(A.intersection(x))/(len_A+len(x)) \
                            if len(x) >= min_length else 0 \
                            for x in idxs_text[i+1:]])
    duplicates = np.argwhere(simularity > threshold / 2) + i + 1
    if len(duplicates) > 0:
        return [i]+[x[0] for x in duplicates]
    else:
        return []  

def job_batch(j, batch_size, idxs_text, min_length, threshold):
    from_ind = j
    to_ind = j + batch_size
    if to_ind > len(idxs_text) - 1:
        to_ind = len(idxs_text) - 1

    pairs = []

    for i in range(from_ind, to_ind):
        A = idxs_text[i]
        if len(A) < min_length:
            continue

        len_A = len(A)
        simularity = np.array([len(A.intersection(x))/(len_A+len(x)) \
                                if len(x) >= min_length else 0 \
                                for x in idxs_text[i+1:]])
        duplicates = np.argwhere(simularity > threshold / 2) + i + 1
        if len(duplicates) > 0:
            pairs.append([i]+[x[0] for x in duplicates])
        else:
            continue

    return pairs
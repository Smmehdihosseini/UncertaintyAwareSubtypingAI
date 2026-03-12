import numpy as np

def determine_majority(row, subtypes):
    counts = row[subtypes].to_dict()
    majority_vote = max(counts, key=counts.get)
    return majority_vote

def calculate_accuracy(cm):

    normalized_values = []
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i])
        diagonal_value = cm[i, i]
        if row_sum > 0:
            normalized_value = diagonal_value / row_sum
            normalized_values.append(normalized_value)

    return np.mean(normalized_values)

def adjust_row(row, subtypes):
    if row['is_certain']==False:
        for column in subtypes:
            if column != row['subtype']:
                row[column] = 0
    return row
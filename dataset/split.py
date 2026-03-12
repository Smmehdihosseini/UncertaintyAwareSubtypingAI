
def update_counts(patient_list, assignment):
    for type_ in patient_list:
        assignment[type_] += 1

def train_val_split(data_df, split_ratio=0.75):

    type_counts = data_df['type'].value_counts()
    train_target = (type_counts * split_ratio).astype(int)
    val_target = type_counts - train_target

    patient_types = data_df.groupby('id')['type'].apply(list)

    train_patients = []
    val_patients = []
    train_counts = {type_: 0 for type_ in type_counts.index}
    val_counts = {type_: 0 for type_ in type_counts.index}

    for patient, types in patient_types.items():
        types_count = {type_: types.count(type_) for type_ in set(types)}

        train_state = sum((train_target[type_] - train_counts[type_]) / train_target[type_] if train_target[type_] else 0 for type_ in types)
        val_state = sum((val_target[type_] - val_counts[type_]) / val_target[type_] if val_target[type_] else 0 for type_ in types)
        
        if train_state >= val_state:
            train_patients.append(patient)
            update_counts(types_count, train_counts)
        else:
            val_patients.append(patient)
            update_counts(types_count, val_counts)

    train_df = data_df[data_df['id'].isin(train_patients)]
    val_df = data_df[data_df['id'].isin(val_patients)]

    return train_df, val_df
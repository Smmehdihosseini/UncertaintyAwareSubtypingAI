from sklearn.model_selection import KFold
import json

def patient_split(ids_df,
                  split_ratio,
                  random_seed,
                  logger,
                  load=True,
                  split_ids_dir=None,
                  default_patients = {"Train":
                                   {
                                       "ccRCC":[],
                                       "pRCC":[],
                                       "CHROMO":[],
                                       "ONCOCYTOMA":[]
                                   },
                                   "Test":
                                   {
                                       "ccRCC":[],
                                       "pRCC":[],
                                       "CHROMO":[],
                                       "ONCOCYTOMA":[]
                                   },
                                   }):

    temp_data_df = ids_df.copy()

    if not load:
        logger.info(">>> Splitting Training/Test IDs ...")
        split_ids = {'Train':{}, 'Test': {}}
        train_test_ids = {'Train':{}, 'Test': {}}

        for subtype, count in split_ratio['Train'].items():
            train_test_ids['Train'][subtype] = []
            train_test_ids['Test'][subtype] = []

            st_temp = (temp_data_df['subtype'] == subtype) & (~temp_data_df['roi_exist'])
            if (st_temp).any():
                for index, _ in ids_df[st_temp].iterrows():
                    train_test_ids['Test'][subtype].append(ids_df.iloc[index]['id'])
                    temp_data_df = temp_data_df.drop(index=index)

            for id in default_patients['Train'][subtype]:
                train_test_ids['Train'][subtype].append(id)
                temp_data_df = temp_data_df.drop(index=temp_data_df[temp_data_df['id']==id].index[0])

            for id in default_patients['Test'][subtype]:
                train_test_ids['Test'][subtype].append(id)
                temp_data_df = temp_data_df.drop(index=temp_data_df[temp_data_df['id']==id].index[0])

            st_df = temp_data_df[temp_data_df['subtype'] == subtype]
            st_train_ids = st_df.sample(n=count-len(train_test_ids['Train'][subtype]), random_state=random_seed)['id'].tolist()
            st_test_ids = [id for id in st_df['id'] if id not in st_train_ids]

            train_test_ids['Train'][subtype].extend(st_train_ids)
            train_test_ids['Test'][subtype].extend(st_test_ids)

            for index in train_test_ids['Train'][subtype]:
                split_ids['Train'][subtype].append(ids_df.iloc[index]['id'])

            for index in train_test_ids['Test'][subtype]:
                split_ids['Test'][subtype].append(ids_df.iloc[index]['id'])

        with open(split_ids_dir, 'w') as file:
            json.dump(split_ids, file, indent=4)

        logger(f">>> Saving Split IDs to '{split_ids_dir}'")

    else:
        try:
            with open(split_ids_dir, 'r') as file:
                split_ids = json.load(file)
            logger(f">>> Loading Split IDs from '{split_ids_dir}'!")
        except FileNotFoundError as e:
            logger.error("Split IDs Couldn't Be Found")
            raise e

    return split_ids

def patient_kfold_split(ids_df,
                        n_folds,
                        random_seed,
                        logger,
                        load=True,
                        split_ids_dir=None):
    
    subtypes = ['ccRCC', 'pRCC', 'CHROMO', 'ONCOCYTOMA']
    fold_splits = {}

    if not load:
        logger.info(">>> Splitting IDs for K-Fold Cross-Validation ...")

        for fold in range(n_folds):
            fold_splits[f'Fold{fold+1}'] = {'Train': {}, 'Test': {}}
            for subtype in subtypes:
                fold_splits[f'Fold{fold+1}']['Train'][subtype] = []
                fold_splits[f'Fold{fold+1}']['Test'][subtype] = []

        for subtype in subtypes:
            st_df = ids_df[ids_df['subtype'] == subtype]
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            for fold, (train_index, test_index) in enumerate(kf.split(st_df)):

                train_ids = st_df.iloc[train_index]['id'].tolist()
                test_ids = st_df.iloc[test_index]['id'].tolist()

                fold_splits[f'Fold{fold+1}']['Train'][subtype].extend(train_ids)
                fold_splits[f'Fold{fold+1}']['Test'][subtype].extend(test_ids)

        with open(split_ids_dir, 'w') as file:
            json.dump(fold_splits, file, indent=4)

        logger.info(f">>> Saving Split IDs to '{split_ids_dir}'")

    else:
        try:
            with open(split_ids_dir, 'r') as file:
                fold_splits = json.load(file)

            logger.info(f">>> Loading Split IDs from '{split_ids_dir}'!")

        except FileNotFoundError as e:
            logger.error("Split IDs Couldn't Be Found")
            raise e

    return fold_splits
import pandas as pd
from sklearn.utils import resample

class Balancer:

    def __init__(self, method='undersample', random_state=42, root_normal_only=False):

        self.method = method
        self.random_state = random_state
        self.root_normal_only = root_normal_only

    def undersample_to_minority(self, df, class_0, class_1, stage):

        if self.root_normal_only and stage == 'Root':
            other_class_0 = [cls for cls in class_0 if cls != 'normal']
            class_0 = ['normal']

        class_0_counts = df[df['type'].isin(class_0)]['type'].value_counts()
        class_1_counts = df[df['type'].isin(class_1)]['type'].value_counts()

        class_0_min_size = min(class_0_counts)
        class_1_min_size = min(class_1_counts)
        
        total_samples_per_group = min(class_0_min_size * len(class_0), class_1_min_size * len(class_1))
        samples_per_class_class_0 = total_samples_per_group // len(class_0)
        samples_per_class_class_1 = total_samples_per_group // len(class_1)
        
        undersampled_class_0 = pd.concat([df[df['type'] == cls].sample(n=samples_per_class_class_0,
                                                                       random_state=self.random_state) for cls in class_0])
        
        undersampled_class_1 = pd.concat([df[df['type'] == cls].sample(n=samples_per_class_class_1,
                                                                       random_state=self.random_state) for cls in class_1])
        
        if self.root_normal_only and stage == 'Root':
            other_class_0_df = df[df['type'].isin(other_class_0)]
            undersampled_df = pd.concat([undersampled_class_0, undersampled_class_1, other_class_0_df]).reset_index(drop=True)
        else:
            undersampled_df = pd.concat([undersampled_class_0, undersampled_class_1]).reset_index(drop=True)
        
        return undersampled_df
        
    def oversample_to_majority(self, df, class_0, class_1, stage):

        if self.root_normal_only and stage == 'Root':
            other_class_0 = [cls for cls in class_0 if cls != 'normal']
            class_0 = ['normal']

        class_0_counts = df[df['type'].isin(class_0)]['type'].value_counts()
        class_1_counts = df[df['type'].isin(class_1)]['type'].value_counts()

        class_0_maj_size = max(class_0_counts)
        class_1_max_size = max(class_1_counts)
        
        total_samples_per_group = max(class_0_maj_size * len(class_0), class_1_max_size * len(class_1))
        samples_per_class_class_0 = total_samples_per_group // len(class_0)
        samples_per_class_class_1 = total_samples_per_group // len(class_1)
        
        oversampled_class_0 = pd.concat([resample(df[df['type'] == cls],
                                                  replace=True, n_samples=samples_per_class_class_0,
                                                  random_state=self.random_state) for cls in class_0])
        
        oversampled_class_1 = pd.concat([resample(df[df['type'] == cls],
                                                  replace=True, n_samples=samples_per_class_class_1,
                                                  random_state=self.random_state) for cls in class_1])
        
        if self.root_normal_only and stage == 'Root':
            other_class_0_df = df[df['type'].isin(other_class_0)]
            oversampled_df = pd.concat([oversampled_class_0, oversampled_class_1, other_class_0_df]).reset_index(drop=True)
        else:
            oversampled_df = pd.concat([oversampled_class_0, oversampled_class_1]).reset_index(drop=True)
        
        return oversampled_df
    
    def apply(self, df, tree_pair_dict):

        balanced_results = {}

        for stage, classes in tree_pair_dict.items():

            class_0 = classes['class_0']
            class_1 = classes['class_1']

            if self.method == 'undersample':
                balanced_df = self.undersample_to_minority(df, class_0, class_1, stage)
            elif self.method == 'oversample':
                balanced_df = self.oversample_to_majority(df, class_0, class_1, stage)

            balanced_results[stage] = balanced_df
            
        return balanced_results
import os
import glob
import re

def find_weights(weights_dir, weights_id, model_params):

    weights = {}
    base_dir = os.path.join(weights_dir, weights_id)

    stage_folders = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ]

    for stage in stage_folders:

        files = glob.glob(os.path.join(stage, '*.weights.h5'))

        if files:
            
            files = [f for f in files if re.search(r'model_epoch_(\d+).weights.h5', f)]
            files.sort(key=lambda x: int(re.search(r'model_epoch_(\d+).weights.h5', x).group(1)),
                        reverse=True)

            latest_update = os.path.basename(files[0])
            stage_name = os.path.basename(stage)

            if model_params[stage_name]['default_weight'] == 'last':
                weights[stage_name] = latest_update
            elif model_params[stage_name]['default_weight'] == 'best':
                weights[stage_name] = 'best_weights.weights.h5'
            else:
                weights[stage_name] = model_params[stage_name]['default_weight']

    return weights
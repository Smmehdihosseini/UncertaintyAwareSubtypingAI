import json
import argparse
from wsi_manager.annotation import WholeSlideAnnotator

try:
    with open("_info/config.json", 'r') as file:
        conf = json.load(file)
except:
    raise FileNotFoundError("Couldn't Load Config File. Check if the `config.json` file exists under `_info` directory.")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Generate Annotations for ExpertDT Results")

    parser.add_argument("--results_dir", type=str, default=conf['results_dir'], help="Results Directory")
    parser.add_argument("--preds_dir", type=str, default=conf['preds_dir'], help="Predictions Files Save Directory")
    parser.add_argument("--res_name", type=str, default=f"result_name",
                        help="Results id name, folder including the patient folders each has .npz files for stages")

    args = parser.parse_args()

    annotator = WholeSlideAnnotator(pred_dir=args.preds_dir, res_dir=args.results_dir, res_name=args.res_name)
    annotator.generate()
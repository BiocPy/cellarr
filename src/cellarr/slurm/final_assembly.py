import json
import sys

from cellarr import CellArrDataset

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def final_assembly(args_json: str):
    args = json.loads(args_json)

    # Perform any final optimizations or validations
    dataset = CellArrDataset(dataset_path=args["output_dir"], assay_uri=args["matrix_names"])

    # Save final metadata
    metadata = {"shape": dataset.shape, "matrices": args["matrix_names"]}

    with open(f"{args['output_dir']}/metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python final_assembly.py '<json_args>'")
        sys.exit(1)

    final_assembly(sys.argv[1])

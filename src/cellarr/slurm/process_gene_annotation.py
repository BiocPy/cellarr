import json

import pandas as pd

from cellarr import utils_anndata as uad
from cellarr.buildutils_tiledb_frame import create_tiledb_frame_from_dataframe

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def process_gene_annotation(args_json: str):
    args = json.loads(args_json)

    # Extract gene information from all files
    files_cache = uad.extract_anndata_info(
        args["files"], var_feature_column=args["gene_options"].get("feature_column", "index")
    )

    # Scan for features
    gene_set = uad.scan_for_features(files_cache)
    gene_set = sorted(gene_set)

    # Create gene annotation dataframe
    gene_annotation = pd.DataFrame({"cellarr_gene_index": gene_set}, index=gene_set)

    # Save gene set for later use
    with open(f"{args['temp_dir']}/gene_set.json", "w") as f:
        json.dump(gene_set, f)

    # Create TileDB store
    create_tiledb_frame_from_dataframe(f"{args['output_dir']}/gene_annotation", gene_annotation)

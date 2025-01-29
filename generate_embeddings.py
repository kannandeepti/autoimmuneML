import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
import spatialdata as sd
from tqdm import tqdm
import h5py

# Import necessary functions from the notebook
# (Make sure these are copied from the notebook to your environment)
from resnet50_plus_ridge import (
    preprocess_spatial_transcriptomics_data_train,
    inf_encoder_factory,
    generate_embeddings,
    read_assets_from_h5,
    load_adata,
)

def create_combined_dataset(
    data_directory_path: str,
    output_directory: str,
    model_weights_path: str,
    size_subset: int = None,
    target_patch_size: int = 64,
):
    """
    Creates a combined dataset containing patches, embeddings, and gene expression data.
    
    Args:
        data_directory_path: Path to input .zarr files
        output_directory: Where to save the processed data
        model_weights_path: Path to ResNet50 weights file
        size_subset: Number of patches to sample per image (optional)
        target_patch_size: Size of image patches
    """
    
    # Create output directories
    os.makedirs(output_directory, exist_ok=True)
    processed_dir = os.path.join(output_directory, "processed_dataset")
    os.makedirs(processed_dir, exist_ok=True)

    # List of datasets to process
    list_ST_name_data = ["UC1_NI", "UC1_I", "UC6_NI", "UC6_I", "UC7_I", "UC9_I", "DC5"]

    # Set up parameters
    args_dict = {
        "size_subset": size_subset,
        "target_patch_size": target_patch_size,
        "show_extracted_images": False,
        "vis_width": 1000,
        "batch_size": 128,
        "num_workers": 0,
        "encoder": "resnet50",
        "weights_root": model_weights_path,
        "overwrite": True
    }
    args = SimpleNamespace(**args_dict)

    # Save training configuration to JSON
    with open(os.path.join(output_directory, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    return None

    # Process images and create patches
    print("Preprocessing spatial transcriptomics data...")
    preprocess_spatial_transcriptomics_data_train(
        list_ST_name_data,
        data_directory_path,
        processed_dir,
        args.size_subset,
        args.target_patch_size,
        args.vis_width,
        args.show_extracted_images
    )

    # Set up for embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = inf_encoder_factory(args.encoder)(args.weights_root)
    embed_dir = os.path.join(processed_dir, "ST_data_emb")
    os.makedirs(embed_dir, exist_ok=True)

    # Load gene list
    gene_path = os.path.join(processed_dir, 'var_genes.json')
    with open(gene_path, 'r') as f:
        genes = json.load(f)['genes']

    # Process each dataset
    combined_data = []
    for name_data in list_ST_name_data:
        print(f"\nProcessing {name_data}...")
        
        # Generate embeddings
        tile_h5_path = os.path.join(processed_dir, "patches", f'{name_data}.h5')
        embed_path = os.path.join(embed_dir, f'{name_data}.h5')
        
        # Generate embeddings if they don't exist
        generate_embeddings(
            embed_path,
            encoder,
            device,
            tile_h5_path,
            args.batch_size,
            args.num_workers,
            overwrite=args.overwrite
        )

        # Load embeddings and metadata
        assets, _ = read_assets_from_h5(embed_path)
        
        # Load patches and their metadata from the patches h5 file
        patches_h5_path = os.path.join(processed_dir, "patches", f'{name_data}.h5')
        with h5py.File(patches_h5_path, 'r') as patches_file:
            patches = patches_file['img'][:]
            patches_barcodes = patches_file['barcode'][:]
            patches_coords = patches_file['coords'][:]

        # Create mapping from barcode to index for patches
        patches_barcode_to_idx = {bc: idx for idx, bc in enumerate(patches_barcodes.flatten().astype(str).tolist())}

        # Load gene expression data
        expr_path = os.path.join(processed_dir, "adata", f'{name_data}.h5ad')
        barcodes = assets['barcodes'].flatten().astype(str).tolist()
        adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize=False)

        # Combine data, ensuring patches match embeddings
        for i in tqdm(range(len(assets['embeddings']))):
            barcode = barcodes[i]
            patch_idx = patches_barcode_to_idx[barcode]
            
            # Verify coordinates match
            assert np.array_equal(assets['coords'][i], patches_coords[patch_idx]), \
                f"Coordinate mismatch for barcode {barcode}"
            
            combined_data.append({
                'sample': name_data,
                'barcode': barcode,
                'patch': patches[patch_idx],  # Use matched patch
                'embedding': assets['embeddings'][i],
                'coords': assets['coords'][i],
                'gene_expression': adata.values[i]
            })

    # Save combined dataset
    print("\nSaving combined dataset...")
    combined_path = os.path.join(output_directory, f"combined_dataset_patch_size_{target_patch_size}.npz")
    np.savez_compressed(
        combined_path,
        samples=[d['sample'] for d in combined_data],
        barcodes=[d['barcode'] for d in combined_data],
        patches=np.stack([d['patch'] for d in combined_data]),
        embeddings=np.stack([d['embedding'] for d in combined_data]),
        coords=np.stack([d['coords'] for d in combined_data]),
        gene_expression=np.stack([d['gene_expression'] for d in combined_data])
    )

    # Save gene names
    gene_names_path = os.path.join(output_directory, "gene_names.json")
    with open(gene_names_path, 'w') as f:
        json.dump({'genes': genes}, f)

    print(f"Dataset saved to {combined_path}")
    print(f"Gene names saved to {gene_names_path}")

if __name__ == "__main__":
    patch_size = int(sys.argv[1])
    # Example usage
    create_combined_dataset(
        data_directory_path='./data',
        output_directory='./embeddings_dataset',
        model_weights_path='./resources/pytorch_model.bin',
        size_subset=None,  # Set to None to use all patches
        target_patch_size=patch_size
    )
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
from torchvision import transforms

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
    encoder_name: str,
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
    
    if encoder_name == 'hoptimus0' and target_patch_size != 518:
        raise ValueError("hoptimus0 expects target patch size of 518 x 518.")
        
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
        "encoder": encoder_name,
        "weights_root": model_weights_path,
        "overwrite": True
    }
    args = SimpleNamespace(**args_dict)

    # Check for existing processed files
    patches_dir = os.path.join(processed_dir, "patches")
    adata_dir = os.path.join(processed_dir, "adata")
    gene_path = os.path.join(processed_dir, 'var_genes.json')
    
    # Check if preprocessing was already done
    preprocessing_complete = all([
        os.path.exists(patches_dir),
        os.path.exists(adata_dir),
        os.path.exists(gene_path),
        all(os.path.exists(os.path.join(patches_dir, f'{name}.h5')) 
            for name in list_ST_name_data)
    ])

    preprocessing_complete=False

    if not preprocessing_complete:
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
    else:
        print("Found existing preprocessed data, skipping preprocessing step...")

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
                #'sample': name_data,
                #'barcode': barcode,
                #'patch': patches[patch_idx],  # Use matched patch
                'embedding': assets['embeddings'][i],
                #'coords': assets['coords'][i],
                'gene_expression': adata.values[i]
            })

    # Save combined dataset
    print("\nSaving combined dataset...")
    combined_path = os.path.join(output_directory, f"combined_dataset_patch_size_{target_patch_size}_{encoder_name}.npz")
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

def test_embeddings(enc_name, model_weights_path, patch_size):
    """ Function to test model loading, embedding generation from a fake image tensor """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = inf_encoder_factory(enc_name)(model_weights_path)
    encoder.eval()
    encoder.to(device)

    img = torch.rand(3, patch_size, patch_size) #random tensor
    img = transforms.ToPILImage()(img) #convert to image (input for eval_transforms)
    img = encoder.eval_transforms(img) #convert back to tensor, normalize by mean/std
    img = img.unsqueeze(0).to(device).float()

    with torch.inference_mode():
        if torch.cuda.is_available():  # Use mixed precision only if CUDA is available
            with torch.amp.autocast('cuda', dtype=encoder.precision):
                embeddings = encoder(img)
        else:  # No mixed precision on CPU
            embeddings = encoder(img)

    return embeddings

def save_dataset(output_directory='./embeddings_dataset'):
    """ Create a .npz file combining the embeddings and gene expression data into one file."""

    processed_dir = os.path.join(output_directory, "processed_dataset")
    embed_dir = os.path.join(processed_dir, "ST_data_emb")
    list_ST_name_data = ["UC1_NI", "UC1_I", "UC6_NI", "UC6_I", "UC7_I", "UC9_I", "DC5"]

    # Load gene list
    gene_path = os.path.join(processed_dir, 'var_genes.json')
    with open(gene_path, 'r') as f:
        genes = json.load(f)['genes']

    combined_embeddings = []
    combined_gene_expr = []
    for name_data in list_ST_name_data:
        embed_path = os.path.join(embed_dir, f'{name_data}.h5')
        assets, _ = read_assets_from_h5(embed_path)
        expr_path = os.path.join(processed_dir, "adata", f'{name_data}.h5ad')
        barcodes = assets['barcodes'].flatten().astype(str).tolist()
        adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize=False)
        combined_embeddings.append(assets['embeddings'])
        combined_gene_expr.append(adata.values)
    
    # Save combined dataset
    print("\nSaving combined dataset...")
    combined_path = os.path.join(output_directory, f"combined_dataset_patch_size_518_hoptimus0.npz")
    np.savez_compressed(
        combined_path,
        embeddings=np.vstack(combined_embeddings),
        gene_expression=np.vstack(combined_gene_expr)
    )


if __name__ == "__main__":
    patch_size = int(sys.argv[1])
    create_combined_dataset(
        data_directory_path='./data',
        output_directory='./embeddings_dataset',
        model_weights_path='./resources/hoptimus0/pytorch_model.bin',
        encoder_name='hoptimus0',
        size_subset=None,  # Set to None to use all patches
        target_patch_size=patch_size
    )
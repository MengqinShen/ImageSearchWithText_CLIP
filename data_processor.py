# process_Unsplash_dataset
from pathlib import Path
import clip
import torch
from PIL import Image
import numpy as np
import pandas as pd
import math

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set the path to the photos
def process_data(photo_metadata, file_path, version="lite",batch_size=16):
    # version="lite"
    # batch_size=16
    dataset_version = version  # Use "lite" or "full"
    photos_path = Path(file_path) / dataset_version / "photos"
    # Path where the feature vectors will be stored
    features_path = Path(file_path) / dataset_version / "features"

    # List all JPGs in the folder
    photos_files = list(photos_path.glob("*.jpg"))
    # photos_files = photo_metadata['photo_id']
    total_photos_files_num = len(photos_files)

    # Function that computes the feature vectors for a batch of images
    def compute_clip_features(photos_batch):
        # Load all the photos from the files
        photos = [Image.open(photo_file) for photo_file in photos_batch]

        # Preprocess all photos
        photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

        with torch.no_grad():
            # Encode the photos batch to compute the feature vectors and normalize them
            photos_features = model.encode_image(photos_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)

        # Transfer the feature vectors back to the CPU and convert to numpy
        return photos_features.cpu().numpy()

    #Process all photos
    # Compute how many batches are needed
    batches_num = math.ceil(total_photos_files_num / batch_size)

    # Process each batch
    for i in range(batches_num):
        print(f"Processing batch {i + 1}/{batches_num}")

        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the photos for the current batch
                batch_files = photos_files[i * batch_size: (i + 1) * batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files)
                np.save(batch_features_path, batch_features)

                # Save the photo IDs and description to a CSV file
                # photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
                photo_ids_data = pd.DataFrame(photo_ids_metadata, columns=['photo_id','description'])
                photo_ids_data.to_csv(batch_ids_path, index=False)
            except:
                # Catch problems with the processing to make the process more robust
                print(f'Problem with batch {i}')
    # Merge the features and the photo IDs. The resulting files are features.npy and photo_ids.csv


    # Load all numpy files
    features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    features_file = features_path / "features.npy"
    np.save(features_file, features)

    # Load all the photo IDs
    photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
    photo_ids_file = features_path / "photo_ids.csv"
    photo_ids.to_csv(photo_ids_file, index=False)

    return photo_ids_file,features_file
# generate the files
# computer_and_save_clip_features(version="lite",batch_size=16)
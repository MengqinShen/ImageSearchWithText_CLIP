import clip
import torch
import pandas as pd
import numpy as np



def image_search(photo_ids_file, photo_features_file,search_query,results_count):

    # Load the open CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # Download the Precomputed Data
    photo_ids = pd.read_csv(photo_ids_file)
    photo_ids = list(photo_ids['photo_id'])
    # Load the features vectors
    photo_features = np.load(photo_features_file)

    # Convert features to Tensors: Float32 on CPU and Float16 on GPU
    if device == "cpu":
      photo_features = torch.from_numpy(photo_features).float().to(device)
    else:
      photo_features = torch.from_numpy(photo_features).to(device)
    # Print some statistics
    print(f"Photos loaded: {len(photo_ids)}")
    search_query = search_query
    if len(search_query) == 0:
        print("Please enter your search query")

    def encode_search_query(search_query):
      with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
      # Retrieve the feature vector
      return text_encoded

    def find_best_matches(text_features, photo_features, photo_ids, results_count=5):
      # Compute the similarity between the search query and each photo using the Cosine similarity
      similarities = (photo_features @ text_features.T).squeeze(1)
      # Sort the photos by their similarity score
      best_photo_idx = (-similarities).argsort()
      # Return the photo IDs of the best matches
      return [photo_ids[i] for i in best_photo_idx[:results_count]]


      # Encode the search query
    text_features = encode_search_query(search_query)
      # Find the best matches
    best_photo_ids = find_best_matches(text_features, photo_features, photo_ids, results_count)
    return best_photo_ids


# search_query = "Two birds flying above the water"
#
# search_unslash(search_query, photo_features, photo_ids, 3)
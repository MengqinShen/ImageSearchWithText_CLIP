# To install the project's dependencies (including those for CLIP) run the following in the terminal. It is a good idea to create a virtual environment.
# pip install -r requirements.txt
# Clone the CLIP repository and copy the code
# Clone the CLIP repository
# !git clone https://github.com/openai/CLIP.git
#
# # Move the CLIP source files and the vocabulary in the root directory.
# # Unfortunately, the CLIP code is not organized as a module, so it cannot be imported easily
# !mv CLIP/*.py .
# !mv CLIP/*.gz .
from data_downloader import download_data
from data_processor import process_data
from model_image_search import image_search
from gemini_ranker import gemini_rank
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def main():

    file_path = "data"
    version = "lite"
    feature_file = f"{file_path}/{version}/features/features.npy"
    if not Path(feature_file).exists():
        photo_metadata = download_data(version="lite", data_path="data",threads_count=16)
        photo_ids_file, photo_features_file = process_data(photo_metadata,file_path, version="lite", batch_size=16)
    else:
        photo_ids_file = "data/lite/features/photo_ids.csv"
        photo_features_file = "data/lite/features/features.npy"
    results_count = 10
    results_count_final = 4
    search_query = input("What picture you want to search?")
    # search_query = "two birds flying"
    best_photo_ids_raw = image_search(photo_ids_file, photo_features_file, search_query, results_count)
    image_ids = gemini_rank(best_photo_ids_raw,file_path,version,search_query,results_count_final)
    # print(best_photo_ids_raw)

    def display_photo(best_photo_ids,image_ids, rows=2):
        if len(best_photo_ids) == 0 or len(image_ids) == 0:
            print("No images found!")
        else:
            display_rows = rows
            display_cols = math.ceil(len(image_ids) / display_rows)
            plt.figure(figsize=(12, rows * 3))
            # for i, photo_id in enumerate(best_photo_ids):
            for i, image_id in enumerate(image_ids):
                photo_id = best_photo_ids[image_id-1]
                print(photo_id)
                photo_image_path = f"{file_path}/{version}/photos/{photo_id}.jpg"
                img = mpimg.imread(photo_image_path)
                plt.subplot(display_rows, display_cols, i + 1)
                plt.imshow(img)
                plt.title("Photo ID :" + photo_id, fontsize=8)
                plt.axis("off")
            plt.suptitle(f'Photo Image found for "{search_query}"', fontsize=16)
            plt.tight_layout()
            plt.show()

    display_photo(best_photo_ids_raw, image_ids, rows=2)
    # display_photo(best_photo_ids_final, rows=2)

if __name__ == "__main__":
    main()

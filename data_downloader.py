# Download the Unsplash dataset
# to download all images from the Unsplash dataset: https://github.com/unsplash/datasets. There are two versions Lite (25000 images) and Full (2M images).
# Put the .TSV files in the folder data/full or data/lite or adjust the path in the cell below.
from pathlib import Path
import pandas as pd
import urllib.request
from multiprocessing.pool import ThreadPool

def download_data(version="lite", data_path="data",threads_count=8):
    dataset_version = version # either "lite" or "full"
    unsplash_dataset_path = Path(data_path) / dataset_version

# Load the dataset
# The photos.tsv000 contains metadata about the photos in the dataset, but not the photos themselves. We will use the URLs of the photos to download the actual images.
# Read the photos table
    photos = pd.read_csv(unsplash_dataset_path / "photos.tsv000", sep='\t', header=0)

    # Extract the IDs and the URLs of the photos
    # photo_metadata = photos[['photo_id', 'photo_description']]

    photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()

    # Print some statistics
    print(f'Photos in the dataset: {len(photo_urls)}')
    # The file name of each photo corresponds to its unique ID from Unsplash. photos downloaded with a reduced resolution (640 pixels width), because they are downscaled by CLIP anyway.
    photos_donwload_path = unsplash_dataset_path / "photos"
    if not photos_donwload_path.exists():
        photos_donwload_path.mkdir(parents=True, exist_ok=True)
        print("📁 Folder created:", photos_donwload_path)
    else:
        print("✅ Folder already exists:", photos_donwload_path)
    # Function that downloads a single photo
    def download_photo(photo):
        # Get the ID of the photo
        photo_id = photo[0]

        # Get the URL of the photo (setting the width to 640 pixels)
        photo_url = photo[1] + "?w=640"

        # Path where the photo will be stored
        photo_path = photos_donwload_path / (photo_id + ".jpg")

        # Only download a photo if it doesn't exist
        if not photo_path.exists():
            try:
                urllib.request.urlretrieve(photo_url, photo_path)
            except:
                # Catch the exception if the download fails for some reason
                print(f"Cannot download {photo_url}")
                # photo_metadata =  photo_metadata[photo_metadata['photo_id'] != photo_id]

                pass

# Now the actual download! The download can be parallelized very well, so we will use a thread pool. You may need to tune the threads_count parameter to achieve the optimzal performance based on your Internet connection. For me even 128 worked quite well.
# Create the thread pool
    threads_count = threads_count
    pool = ThreadPool(threads_count)
    # Start the download
    pool.map(download_photo, photo_urls)

    # Display some statistics
    print(f'Photos downloaded: {len(photos)} to {unsplash_dataset_path}')

    # return photo_metadata
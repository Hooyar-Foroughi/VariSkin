import os
import zipfile
import gdown
import requests
import urllib.request
import json
from pathlib import Path
import pandas as pd
import concurrent.futures

# Get absolute path of src/ directory (move up one level from utils/)
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # src/
METRICS_FILE = os.path.join(SRC_DIR, "metrics.csv")
ARCHIVE_ZIP = os.path.join(SRC_DIR, "archive.zip")
ARCHIVE_FOLDER = os.path.join(SRC_DIR, "ISIC_Archive")

# Google Drive URLs
METRICS_URL = "https://drive.google.com/uc?id=1XaWddknL_xssWLKbgm4RRhAkoP1tZp5r"
ARCHIVE_URL = "https://drive.google.com/uc?id=1EhFl9JD8NwpUgh37veK1XLB52nYUYkn2"

# ISIC API base URL
BASE_URL = "https://api.isic-archive.com/api/v2/images/"


def get_metadata():
    # Create metadata directory if it doesn't exist
    metadata_dir = Path(ARCHIVE_FOLDER, "metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Read metrics.csv to get ISIC IDs
    print("Reading metrics.csv...")
    df = pd.read_csv(METRICS_FILE)
    # Remove .jpg extension from img_name
    isic_ids = df["img_name"].str.replace(".jpg", "").tolist()  

    # Define a function to download metadata for one ID
    def _download_metadata_for_id(isic_id):
        metadata_path = metadata_dir / f"{isic_id}.json"
        if not metadata_path.exists():
            try:
                get_image_by_isic_id(
                    base_url=BASE_URL,
                    isic_id=isic_id,
                    output_dir=ARCHIVE_FOLDER,
                    download_images=False,
                    download_metadata=True,
                )
            except Exception as e:
                print(f"Error downloading metadata for {isic_id}: {e}")

    # Download metadata in parallel using ThreadPoolExecutor
    print("Fetching metadata files if needed (this may take a few minutes)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(_download_metadata_for_id, isic_ids)


def get_image_by_isic_id(
    base_url: str,
    isic_id: str,
    output_dir: str,
    download_images: bool,
    download_metadata: bool,
) -> None:
    """Download an image and/or metadata by its ISIC ID.

    Args:
        base_url: The base URL of the ISIC API.
        isic_id: The ISIC ID of the image to download.
        output_dir: The directory to save the files to.
        download_images: Whether to download the image.
        download_metadata: Whether to download the metadata.

    Returns:
        None
    """
    # Create the URL to download the image metadata from
    img_url = f"{base_url}{isic_id}"

    # Get the JSON response from the API
    response = requests.get(img_url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Parse the JSON response
    json_data = response.json()

    if download_images:
        # Get the image URL from the JSON response
        img_url = json_data["files"]["full"]["url"]
        img_path = Path(output_dir) / f"images/{isic_id}.jpg"
        urllib.request.urlretrieve(img_url, img_path)

    if download_metadata:
        # Check if metadata exists in the response
        if "metadata" in json_data:
            metadata = json_data["metadata"]
            metadata_path = Path(output_dir) / f"metadata/{isic_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            print(f"Metadata not found for {isic_id}")


def download_data():
    """Download and extract necessary data files into src/ directory, including metadata."""

    # Download metrics.csv into src/
    if not os.path.exists(METRICS_FILE):
        print("Downloading metrics.csv...")
        gdown.download(METRICS_URL, METRICS_FILE, quiet=False)

    # Download and extract archive.zip into src/
    if not os.path.exists(ARCHIVE_FOLDER):
        if not os.path.exists(ARCHIVE_ZIP):
            print("Downloading archive.zip...")
            gdown.download(ARCHIVE_URL, ARCHIVE_ZIP, quiet=False)

        print("Extracting archive.zip...")
        with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zip_ref:
            zip_ref.extractall(ARCHIVE_FOLDER)  # Extract inside src/


if __name__ == "__main__":
    download_data()

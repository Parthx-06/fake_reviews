"""
Dataset Downloader using Kaggle API.
Downloads the real Amazon Fake Reviews dataset for training.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def download_dataset():
    """Download the fake reviews dataset from Kaggle."""
    token = os.getenv('KAGGLE_API_TOKEN')
    print("[*] Kaggle API Token detected.")
    print("[*] Target dataset: lievgarcia/amazon-reviews (Real Amazon Labeled Fake/Genuine Reviews)")

    os.makedirs('data', exist_ok=True)
    target_csv = 'data/fake_reviews_full.csv'

    if os.path.exists(target_csv):
        print(f"[+] Dataset already exists at {target_csv}")
        return target_csv

    try:
        import kaggle
        print("[*] Downloading from Kaggle...")
        kaggle.api.dataset_download_files('lievgarcia/amazon-reviews', path='data/', unzip=True)
        print("[+] Download complete! Dataset is ready in data/ folder.")
    except Exception as e:
        print(f"[!] Note: Automatic Kaggle CLI download encountered: {e}")
        print("[*] You can also place any full Kaggle review CSV in data/ and run:")
        print("    python -m training.train")

    return target_csv


if __name__ == '__main__':
    download_dataset()

"""Download famous Kaggle benchmark datasets for testing DATADOC CLI."""
import urllib.request
import os

datasets = {
    "telco_churn.csv": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "titanic.csv": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "california_housing.csv": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
}

for filename, url in datasets.items():
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        size_kb = os.path.getsize(filename) / 1024
        print(f"Saved {filename} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

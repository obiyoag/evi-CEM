import os
import requests
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def download_image(row):
    hash, url = row
    try:
        response = requests.get(url, headers={"User-Agent": "Wget/1.21.1 (linux-gnu)"})
        response.raise_for_status()
        with open(f"data/raw_data/{hash}.jpg", "wb") as file:
            file.write(response.content)
    except requests.exceptions.HTTPError as err:
        print(err)


if __name__ == "__main__":
    os.makedirs("data/raw_data", exist_ok=True)
    df = pd.read_csv("data/meta_data/fitzpatrick17k.csv")
    print(f"{len(df[df['url'].isna()])} samples do not have url")
    df = df[df["url"].notna()]  # filter out samples without the url
    print(f"remaining {len(df)} samples")

    total_rows = len(df)
    # create thread pool and download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor, tqdm(total=total_rows, desc="Downloading") as pbar:
        futures = []
        for row in df[["md5hash", "url"]].itertuples(index=False):
            future = executor.submit(download_image, row)
            futures.append(future)

        # update processor bar
        for future in futures:
            future.result()
            pbar.update(1)

    print(f"we have {len(os.listdir('data/raw_data'))} samples in total")

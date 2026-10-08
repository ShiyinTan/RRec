"""Turn the classic Foursquare NYC / Tokyo check-in data into RRec's dataset format.

The raw files are Dingqi Yang et al., TSMC 2015 (dataset_TSMC2014_NYC.txt and
dataset_TSMC2014_TKY.txt). This script downloads a public zip mirror, applies an
iterative K-core filter, and writes a leave-one-out next-POI split that
``train.py`` can load with ``datasets.load_from_disk``.

Check-ins have no star rating. Every visit is stored with rating 1, and the POI
prompt treats that number as a check-in flag.
"""

import os
import re
import zipfile

import fire
import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from tqdm import tqdm

ZIP_URL = (
    "https://huggingface.co/datasets/Davidham3/FourSquare_NYC_TKY/resolve/main/dataset_tsmc2014.zip"
)

DATASETS = {
    "NYC": {
        "member_suffix": "dataset_TSMC2014_NYC.txt",
        "city": "New York City",
    },
    "TKY": {
        "member_suffix": "dataset_TSMC2014_TKY.txt",
        "city": "Tokyo",
    },
}

COLUMNS = [
    "user_id",
    "venue_id",
    "category_id",
    "category_name",
    "latitude",
    "longitude",
    "timezone_offset",
    "utc_time",
]

TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def clean_text(raw_text: str) -> str:
    text = "" if raw_text is None else str(raw_text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_zip(data_root_dir: str, raw_dirname: str) -> str:
    raw_dir = os.path.join(data_root_dir, raw_dirname)
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "dataset_tsmc2014.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"{zip_path} already exists. Skipping download.")
        return zip_path

    response = requests.get(ZIP_URL, stream=True, timeout=60)
    if response.status_code != 200:
        raise ValueError(f"Failed to download POI data. HTTP status code: {response.status_code}")
    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as fout, tqdm(
        desc="Downloading dataset_tsmc2014.zip",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            fout.write(chunk)
            bar.update(len(chunk))
    return zip_path


def read_checkins(zip_path: str, member_suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(member_suffix))
        with archive.open(member) as handle:
            frame = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=COLUMNS,
                encoding="latin-1",
                dtype={"user_id": str, "venue_id": str, "category_id": str, "category_name": str},
            )
    frame["category_name"] = frame["category_name"].map(clean_text)
    frame["utc_time"] = pd.to_datetime(frame["utc_time"], format=TIME_FORMAT, utc=True)
    frame = frame.dropna(subset=["user_id", "venue_id", "utc_time", "latitude", "longitude"])
    frame = frame.sort_values(["utc_time", "user_id", "venue_id"], kind="mergesort")
    return frame.reset_index(drop=True)


def iterative_k_core(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 1:
        return frame
    current = frame
    while True:
        user_counts = current.groupby("user_id").size()
        venue_counts = current.groupby("venue_id").size()
        keep_users = user_counts[user_counts >= k].index
        keep_venues = venue_counts[venue_counts >= k].index
        filtered = current[current["user_id"].isin(keep_users) & current["venue_id"].isin(keep_venues)]
        if len(filtered) == len(current):
            return filtered.reset_index(drop=True)
        current = filtered


def venue_title(category_name: str, latitude: float, longitude: float, venue_id: str) -> str:
    name = category_name or "Place"
    return f"{name} ({latitude:.5f}, {longitude:.5f}) [{venue_id}]"


def build_records(frame: pd.DataFrame, city: str, window_size: int):
    venue_meta = (
        frame.groupby("venue_id", sort=False)
        .agg(
            category_name=("category_name", "first"),
            category_id=("category_id", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            checkins=("user_id", "size"),
        )
        .reset_index()
    )
    venue_ids = venue_meta["venue_id"].tolist()
    venue_to_item_id = {venue_id: idx + 1 for idx, venue_id in enumerate(venue_ids)}

    item_rows = []
    title_by_item_id = {}
    for row in venue_meta.itertuples(index=False):
        item_id = venue_to_item_id[row.venue_id]
        title = venue_title(row.category_name, row.latitude, row.longitude, row.venue_id)
        title_by_item_id[item_id] = title
        description = (
            f"A {row.category_name or 'place'} in {city}. "
            f"Latitude {row.latitude:.5f}, longitude {row.longitude:.5f}."
        )
        item_rows.append(
            {
                "main_category": "POI",
                "title": title,
                "subtitle": row.category_name,
                "average_rating": 0.0,
                "rating_number": int(row.checkins),
                "features": [],
                "description": [description],
                "price": 0.0,
                "store": city,
                "categories": [row.category_name] if row.category_name else [],
                "details": "",
                "parent_asin": row.venue_id,
                "item_id": item_id,
            }
        )

    item_rows.append(
        {
            "main_category": "POI",
            "title": "pad_title",
            "subtitle": "pad_subtitle",
            "average_rating": 0.0,
            "rating_number": 0,
            "features": [],
            "description": [],
            "price": 0.0,
            "store": "pad_store",
            "categories": [],
            "details": "",
            "parent_asin": "pad_asin",
            "item_id": 0,
        }
    )

    splits = {"train": [], "valid": [], "test": []}
    for user_id, user_frame in frame.groupby("user_id", sort=False):
        user_frame = user_frame.sort_values("utc_time", kind="mergesort")
        venues = user_frame["venue_id"].tolist()
        timestamps = [int(ts.timestamp() * 1000) for ts in user_frame["utc_time"]]
        if len(venues) < 3:
            continue
        item_ids = [venue_to_item_id[venue_id] for venue_id in venues]
        titles = [title_by_item_id[item_id] for item_id in item_ids]
        for index in range(1, len(venues)):
            start = max(index - window_size, 0)
            if index == len(venues) - 1:
                split = "test"
            elif index == len(venues) - 2:
                split = "valid"
            else:
                split = "train"
            splits[split].append(
                {
                    "user_id": str(user_id),
                    "item_asins": venues[start:index],
                    "item_asin": venues[index],
                    "history_item_id": item_ids[start:index],
                    "item_id": item_ids[index],
                    "history_item_title": titles[start:index],
                    "item_title": titles[index],
                    "history_rating": [1] * (index - start),
                    "rating": 1,
                    "history_timestamp": timestamps[start:index],
                    "timestamp": timestamps[index],
                }
            )

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: row["timestamp"])
        for interaction_id, row in enumerate(split_rows):
            row["interaction_id"] = interaction_id

    return item_rows, splits


def validate_alignment(dataset_dict: DatasetDict):
    item_info = dataset_dict["item_info"]
    real_count = len(item_info) - 1
    for index in range(real_count):
        if int(item_info[index]["item_id"]) != index + 1:
            raise ValueError(f"item_info row {index} has item_id {item_info[index]['item_id']}")
    if int(item_info[-1]["item_id"]) != 0:
        raise ValueError("The padding POI must be the last item_info row and use item_id 0.")

    max_history = 0
    for split in ("train", "valid", "test"):
        table = dataset_dict[split]
        if len(table) == 0:
            raise ValueError(f"Split {split} is empty.")
        for row_index in range(min(len(table), 20)):
            row = table[row_index]
            item_id = int(row["item_id"])
            looked_up = item_info[item_id - 1]
            if looked_up["title"] != row["item_title"]:
                raise ValueError(
                    f"{split} row {row_index} title does not match item_info[{item_id - 1}]"
                )
            for history_id, history_title in zip(row["history_item_id"], row["history_item_title"]):
                if item_info[int(history_id) - 1]["title"] != history_title:
                    raise ValueError(f"{split} row {row_index} has a mismatched history title")
        max_history = max(max_history, max(len(ids) for ids in table["history_item_id"]))
    return max_history


def convert_one(zip_path: str, dataset: str, k: int, window_size: int, data_root_dir: str) -> str:
    spec = DATASETS[dataset]
    print(f"Reading {dataset} check-ins")
    frame = read_checkins(zip_path, spec["member_suffix"])
    print(f"Raw {dataset}: users={frame['user_id'].nunique()} venues={frame['venue_id'].nunique()} checkins={len(frame)}")
    frame = iterative_k_core(frame, k)
    print(
        f"After {k}-core: users={frame['user_id'].nunique()} "
        f"venues={frame['venue_id'].nunique()} checkins={len(frame)}"
    )

    item_rows, splits = build_records(frame, spec["city"], window_size)
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(splits["train"]),
            "valid": Dataset.from_list(splits["valid"]),
            "test": Dataset.from_list(splits["test"]),
            "item_info": Dataset.from_list(item_rows),
        }
    )
    max_history = validate_alignment(dataset_dict)
    if window_size > max_history:
        print(
            f"Longest saved history is {max_history}. "
            f"Train with --window_size less than or equal to {max_history}."
        )

    out_dir = os.path.join(data_root_dir, f"{dataset}_{k}")
    dataset_dict.save_to_disk(out_dir)
    print(
        f"Saved {out_dir}: train={len(dataset_dict['train'])}, "
        f"valid={len(dataset_dict['valid'])}, test={len(dataset_dict['test'])}, "
        f"items={len(dataset_dict['item_info'])}, max_history={max_history}"
    )
    return out_dir


def main(
    dataset: str = "NYC",
    K: int = 10,
    window_size: int = 20,
    data_root_dir: str = "data",
    raw_dirname: str = "raw_poi",
):
    """Prepare Foursquare NYC and/or Tokyo for RRec.

    Args:
        dataset: NYC, TKY, or all.
        K: Minimum check-ins for a user and a place. Users and places below K
            are removed iteratively.
        window_size: Maximum number of previous visits kept as history.
        data_root_dir: Directory for the raw zip and the processed dataset.
    """
    names = list(DATASETS) if dataset == "all" else [dataset]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset {unknown}. Choose from {list(DATASETS)} or all.")
    if K < 0:
        raise ValueError("K must be >= 0. Leave-one-out still requires users with at least 3 visits.")
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    zip_path = download_zip(data_root_dir, raw_dirname)
    for name in names:
        convert_one(zip_path, name, K, window_size, data_root_dir)


if __name__ == "__main__":
    fire.Fire(main)

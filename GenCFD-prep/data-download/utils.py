import os
import requests
import re
import pandas as pd
from tqdm import tqdm


def extract_velocity_links_from_toc_html(
    raw_toc_fpath: str,
    output_fpath: str,
    fname_pattern: str = r"velocity[0-9]+-[0-9]+\.feather",
) -> None:
    """
    This function parses the HTML table of contents to extract download information
    for velocity files matching the specified pattern, and saves the results as a CSV.
    The table of contents can be found at `https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/LAI6YU&version=2.0`
    (or version=1.0) and its raw HTML has been saved in the `raw_table_of_contents_V1-velocity.txt` and `raw_table_of_contents_V2-velocity.txt` files.

    Args:
        raw_toc_fpath: Path to the raw HTML table of contents file.
        output_fpath: Path where the extracted links will be saved as CSV.
        fname_pattern: Regex pattern to match velocity filenames.
    """
    # Read the file
    with open(raw_toc_fpath, "r") as f:
        content = f.read()

    # Find all links with the pattern
    pattern = (
        r'<a href="/file\.xhtml\?persistentId=doi:10\.57745/[A-Z0-9]+&amp;version=[12]\.0">\s*'
        + fname_pattern
        + r"\s*</a>"
    )
    matches = re.findall(pattern, content)
    print(f"[INFO] Found {len(matches)} matches.")

    df = pd.DataFrame(matches, columns=["link_element"])
    df["persistent_id"] = df["link_element"].str.extract(
        r"persistentId=doi:10\.57745/([A-Z0-9]+)"
    )
    df["fname"] = df["link_element"].str.extract(r"(" + fname_pattern + r")")
    df["download_link"] = df["persistent_id"].apply(
        lambda x: f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId?persistentId=doi:10.57745/{x}"
    )
    df.to_csv(output_fpath, index=False)


def download_velocity_files(
    download_urls_fpath: str, save_dir: str, max_downloads: int = None
) -> None:
    """
    This function reads the CSV containing download URLs, checks for already downloaded files,
    and downloads the remaining files with progress tracking. It handles errors by removing
    partially downloaded files.

    Args:
        download_urls_fpath: Path to CSV file containing download URLs.
        save_dir: Directory where downloaded files will be saved.
        max_downloads: Maximum number of files to download (None for all).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load download URLs
    df = pd.read_csv(download_urls_fpath)

    # Check for files already downloaded
    existing_files = set(os.listdir(save_dir))
    df["already_downloaded"] = df["fname"].isin(existing_files)

    print(
        f"[INFO] Found {len(df)} files to download. "
        f'{df["already_downloaded"].sum()} files already downloaded. '
        f'Downloading the remaining {(~df["already_downloaded"]).sum()} files.'
    )

    df = df[~df["already_downloaded"]]
    if max_downloads is not None:
        df = df.head(max_downloads)
        print(
            f"[INFO] Max downloads set to {max_downloads}. Downloading only the first {len(df)} files."
        )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading files"):
        url, fname = row[["download_link", "fname"]]
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(os.path.join(save_dir, fname), "wb") as f:
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            except Exception as e:
                print(f"[ERROR] Error downloading file {fname}: {e}")
                os.remove(os.path.join(save_dir, fname))

import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

ONTOLOGIES = { # https://obofoundry.org/
    "pr": "https://purl.obolibrary.org/obo/pr.obo",   # Protein Ontology
    "doid": "https://purl.obolibrary.org/obo/doid.obo",  # Disease Ontology
    "uberon": "https://purl.obolibrary.org/obo/uberon/uberon-basic.obo",  # Uberon Anatomy Ontology
    "cl": "https://purl.obolibrary.org/obo/cl/cl-basic.obo",  # Cell Line Ontology
    "ncbitaxon": "https://purl.obolibrary.org/obo/ncbitaxon.obo",  # NCBI Taxonomy
}

PARENT_DIR = Path(__file__).parents[1] / "data" / "ontology"


def get_path(url) -> Path:
    """Create output path based on the URL."""
    return PARENT_DIR / Path(urlparse(url).path).name


def download_file(url, output_path):
    """Download with streaming and show progress bar."""
    with requests.get(url, stream=True) as response:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error occurred while downloading {url}: {e}")
            return

        file_size = int(response.headers.get('content-length', 0))
        with tqdm(total=file_size, unit='B', unit_scale=True) as bar:
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download ontology files.")
    parser.add_argument("names", nargs="*", choices=list(ONTOLOGIES), help="ontologies to download")
    parser.add_argument("--all", action="store_true", help="download all ontologies")
    args = parser.parse_args()

    names = list(ONTOLOGIES) if args.all else args.names
    if not names:
        parser.error("specify at least one ontology or --all")

    PARENT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        url = ONTOLOGIES[name]
        print(f"Downloading {name} from {url}")
        download_file(url, get_path(url))


if __name__ == "__main__":
    main()

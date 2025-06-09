import os
import argparse


from utils import extract_velocity_links_from_toc_html, download_velocity_files


def main() -> None:
    """
    Main function for managing HemewS-3D velocity data acquisition.

    Handles command-line interface for extracting download links and downloading
    velocity files across multiple dataset versions.
    """
    parser = argparse.ArgumentParser(
        description="Extract links and download velocity files from HemewS-3D dataset"
    )
    parser.add_argument(
        "--extract", action="store_true", help="Extract links from table of contents"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download velocity files"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
        help="Versions to process (1, 2, or both)",
    )
    parser.add_argument(
        "--metadata-dirpath",
        default="./",
        help="Directory where metadata files (TOC and URL files) are stored",
    )
    parser.add_argument(
        "--rawdata-dirpath",
        default="/cluster/work/math/camlab-data/Wave_HemewS-3D",
        help="Base directory to save downloaded data files",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Maximum number of files to download (None for all)",
    )
    args = parser.parse_args()

    # Process for each requested version
    for version in args.versions:
        toc_filename = f"raw_table_of_contents_V{version}-velocity.txt"
        urls_filename = f"download_urls_V{version}-velocity.csv"
        toc_filepath = os.path.join(args.metadata_dirpath, toc_filename)
        urls_filepath = os.path.join(args.metadata_dirpath, urls_filename)
        save_dir = os.path.join(args.rawdata_dirpath, f"version{version}", "velocity")

        # Pattern depends on the version
        fname_pattern = (
            r"velocity[0-9]+-[0-9]+\.feather"
            if version == 1
            else r"velocity[0-9]+-[0-9]+\.zip"
        )

        # Extract links if requested
        if args.extract:
            print(f"[INFO] Extracting links for version {version}")
            extract_velocity_links_from_toc_html(
                raw_toc_fpath=toc_filepath,
                output_fpath=urls_filepath,
                fname_pattern=fname_pattern,
            )

        # Download files if requested
        if args.download:
            print(f"[INFO] Downloading files for version {version}")
            download_velocity_files(
                download_urls_fpath=urls_filepath,
                save_dir=save_dir,
                max_downloads=args.max_downloads,
            )

    # If no action specified, show help
    if not (args.extract or args.download):
        parser.print_help()


if __name__ == "__main__":
    main()

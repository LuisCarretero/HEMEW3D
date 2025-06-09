

- Preprocessing pipeline for the HemewS-3D dataset to prepare it for training using GenCFD-derived models.

- Data-download details:
    - Data is hosted on https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/LAI6YU&version=1.0. Downloading entire dataset is cumbersome to say the least.
    - Its assumed that underground velocity fields are already downloaded and available in the `data-download` directory (they are rather small and can be downloaded in one go using curl).
    - The velocity traces can be downloaded using the `download_all.py` script. For this the raw HTML of the table from the hosting website is needed (already provided in the `data-download` directory). This is then parsed to extract the download links and the corresponding file names.
    - The velocity tracefiles are then downloaded and saved in the `data-download` directory.

- Preprocessing pipeline details:
    - The velocity traces are processed to prepare them for training using GenCFD-derived models.
    - Add implementation details and options. 
    - Mention metadata files written to dir which is used at trainign time to infer data shapes, etc.
[![Documentation Status](https://readthedocs.org/projects/photonlib/badge/?version=latest)](https://photonlib.readthedocs.io/en/latest/)

# PhotonLib

PhotonLib is a small PyTorch-friendly API for loading and querying detector
photon libraries. A photon library stores, for every voxel in a detector
volume, the probability that a photon reaches each optical detector. Looking
up those precomputed values is much faster than transporting every photon in a
Monte Carlo simulation.

The package supports dense in-memory libraries and lazy HDF5 access for tables
that are too large to materialize. Visibility records may contain a traditional
per-detector vector or higher-dimensional payloads such as detector waveforms.

## Installation

PhotonLib requires Python 3.10 or newer. Install a checkout with:

```bash
python -m pip install .
```

For an editable development install with the test runner:

```bash
python -m pip install -e . pytest
pytest -q
```

## Quick start

```python
import torch
from photonlib import PhotonLib

# lazy=True keeps the HDF5 visibility dataset on disk until rows are queried.
library = PhotonLib.load("plib.h5", lazy=True)

positions = torch.tensor([
    [10.0, 20.0, 30.0],
    [15.0, 25.0, 35.0],
])
visibility = library.visibility(positions)

print(library.meta)       # voxelization and coordinate bounds
print(visibility.shape)   # (positions, optical detectors, ...)
```

Coordinates are expressed in the absolute coordinate system stored in the
file. Queries outside that volume return zeros. Calling `PhotonLib.load()`
without `lazy=True` loads the complete visibility table into memory.

A compatible HDF5 file contains these datasets:

- `numvox`: voxel counts along each spatial axis
- `min` and `max`: spatial bounds
- `vis`: visibility payload with voxels on axis 0
- `eff` (optional): global efficiency scale

## Example data

The installed helper scripts download public example libraries into the
current directory:

```bash
download_icarus_plib.sh   # writes plib_icarus.h5
download_2x2_plib.sh      # writes three 2x2 library files
```

Additional sample material is available in the
[project Google Drive folder](https://drive.google.com/drive/folders/1IjRUMMVW7aiGWGcZFGRb9nT8dCRVYolE?usp=share_link).

## Development

Run the full suite with `pytest -q`. See [contributing.md](contributing.md) for
the repository workflow and the
[Read the Docs site](https://photonlib.readthedocs.io/) for API documentation.

PhotonLib is released under the [MIT License](LICENSE.md).

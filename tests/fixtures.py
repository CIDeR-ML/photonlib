import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest
import torch
from photonlib import PhotonLib
from tests.conftest import GLOBAL_SEED

@pytest.fixture
def rng(GLOBAL_SEED):
    return  np.random.default_rng(GLOBAL_SEED)

@pytest.fixture
def torch_rng(GLOBAL_SEED):
    return torch.Generator().manual_seed(GLOBAL_SEED)



@pytest.fixture
def num_pmt():
    """can't change because of hardcoded values in the SIREN model"""
    return 180

def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name

@pytest.fixture
def fake_photon_library_ranges():
    return [[-400, -35], 
            [-200, 170],
            [-1000, 1000]]
    
@pytest.fixture
def fake_photon_library_shape():
    return (10, 25, 30)

@pytest.fixture
def fake_photon_library(rng, num_pmt, fake_photon_library_ranges, fake_photon_library_shape):
    """
    h5 file has the following structure:
       - numvox: number of voxels in each dimension with shape (3,)
       - vis: 3D array of visibility values with shape (numvox, Npmt)
       - min: minimum coordinate of the active volume with shape (3,)
       - max: maximum coordinate of the active volume with shape (3,)
    """
    fake_h5 = writable_temp_file(suffix='.h5')
    with h5py.File(fake_h5, 'w') as f:
        f.create_dataset('numvox', shape=(3,), data=fake_photon_library_shape)
        total_numvox = np.prod(f['numvox'][:])

        # fake vis data -- random numbers uniformly distributed from 10^-7 to 10^-3
        vis = 10**rng.uniform(low=-7, high=-3, size=(total_numvox, num_pmt))
        f.create_dataset('vis', shape=(total_numvox, num_pmt), data=vis)
        
        # fake min/max data
        mins, maxs = np.array(fake_photon_library_ranges).T
        
        f.create_dataset('min', shape=(3,), data=mins)
        f.create_dataset('max', shape=(3,), data=maxs)
    yield fake_h5
    os.remove(fake_h5)

@pytest.fixture
def plib(fake_photon_library):
    return PhotonLib.load(fake_photon_library)

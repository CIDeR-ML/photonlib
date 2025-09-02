import h5py
import torch
import numpy as np

from photonlib import PhotonLib, VoxelMeta
from photonlib.lazy import LazyTensor
from tests.fixtures import fake_photon_library, fake_photon_library_ranges as ranges_fixture, num_pmt


def test_photonlib_load_lazy_returns_lazytensor(fake_photon_library, num_pmt):
    plib = PhotonLib.load(fake_photon_library, lazy=True)
    assert isinstance(plib.vis, LazyTensor)

    with h5py.File(fake_photon_library, 'r') as f:
        ds = f['vis']
        assert len(plib) == ds.shape[0]
        assert tuple(plib.vis.shape) == tuple(ds.shape)
        assert plib.n_pmts == num_pmt


def test_photonlib_lazy_indexing_matches_dataset(fake_photon_library):
    plib = PhotonLib.load(fake_photon_library, lazy=True)

    # compare a few rows to the dataset directly
    f = h5py.File(fake_photon_library, 'r', swmr=True, libver='latest')
    ds = f['vis']
    for i in [0, len(plib)-1, len(plib)//2]:
        assert torch.allclose(plib[i], torch.as_tensor(ds[i], dtype=torch.float32))


def test_photonlib_visibility_lazy_matches_expected(fake_photon_library, ranges_fixture):
    # use a few random positions inside the volume; compare to dataset rows
    plib = PhotonLib.load(fake_photon_library, lazy=True)
    meta = plib.meta

    tranges = torch.as_tensor(ranges_fixture)
    # 5 random points inside the ranges
    torch.manual_seed(0)
    pos = torch.rand(5, 3) * (tranges[:, 1] - tranges[:, 0]) + tranges[:, 0]

    vox = meta.coord_to_voxel(pos)

    # expected vis by reading dataset rows for those voxels
    # h5py requires increasing order for fancy indexing; fetch row-by-row
    with h5py.File(fake_photon_library, 'r') as f:
        ds = f['vis']
        vox_np = vox.cpu().numpy().tolist()
        rows = [torch.as_tensor(ds[i], dtype=torch.float32) for i in vox_np]
        expected = torch.stack(rows, dim=0)

    out = plib.visibility(pos)
    assert torch.allclose(out, expected)


def test_photonlib_gradx2_lazy_consistency_with_dense(fake_photon_library):
    # gradx2 uses LazyTensor slicing and arithmetic; compare with dense path
    plib_lazy = PhotonLib.load(fake_photon_library, lazy=True)
    plib_dense = PhotonLib.load(fake_photon_library, lazy=False)

    # a couple of random positions
    torch.manual_seed(1)
    idx = torch.randint(low=0, high=len(plib_lazy), size=(4,))
    pos = plib_lazy.meta.voxel_to_coord(idx)

    g_lazy = plib_lazy.gradx2(pos)
    g_dense = plib_dense.gradx2(pos)

    assert torch.allclose(g_lazy, g_dense)


def test_photonlib_to_same_device_lazy_identity(fake_photon_library):
    # In lazy mode on CPU, to('cpu') should return self
    plib = PhotonLib.load(fake_photon_library, lazy=True)
    plib2 = plib.to('cpu')
    assert plib2 is plib
    # sanity: visibility still works
    idx = torch.tensor([0, len(plib)//2])
    pos = plib.meta.voxel_to_coord(idx)
    out = plib2.visibility(pos)
    assert out.shape[0] == 2

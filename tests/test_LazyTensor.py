import gc
import h5py
import numpy as np
import torch

from photonlib.lazy import LazyTensor
from tests.fixtures import fake_photon_library


def test_lazytensor_from_tensor_cpu():
    src = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    lt = LazyTensor(src)

    assert lt.device.type == 'cpu'
    assert lt.dtype == torch.float32
    assert len(lt) == 10
    assert tuple(lt.shape) == (10, 2)

    # indexing and materialize should match source
    assert torch.allclose(lt[3], src[3])
    assert torch.allclose(lt[0:4], src[0:4])
    mat = lt.materialize()
    assert isinstance(mat, torch.Tensor)
    assert torch.allclose(mat, src)

    # to on same device returns self
    lt2 = lt.to('cpu')
    assert lt2 is lt


def test_lazytensor_from_h5_dataset(fake_photon_library):
    # open file without context manager so LazyTensor manages closing
    f = h5py.File(fake_photon_library, 'r', swmr=True, libver='latest')
    ds = f['vis']

    lt = LazyTensor(ds, dtype=torch.float32)
    assert lt.device.type == 'cpu'
    assert lt.dtype == torch.float32
    assert len(lt) == ds.shape[0]
    assert tuple(lt.shape) == tuple(ds.shape)

    # spot check indexing and slicing
    assert torch.allclose(lt[0], torch.as_tensor(ds[0], dtype=torch.float32))
    assert torch.allclose(lt[len(lt)-1], torch.as_tensor(ds[len(lt)-1], dtype=torch.float32))

    sl = slice(1, 5)
    assert torch.allclose(lt[sl], torch.as_tensor(ds[sl], dtype=torch.float32))

    # materialize should match full dataset
    mat = lt.materialize()
    assert isinstance(mat, torch.Tensor)
    assert torch.allclose(mat, torch.as_tensor(ds[:], dtype=torch.float32))

    # deletion should close the file the dataset came from
    del lt
    gc.collect()
    # file may already be closed by LazyTensor; check "valid" if available
    try:
        assert not f.id.valid
    except Exception:
        # if attribute access not available, at least ensure operations fail
        failed = False
        try:
            _ = ds.shape  # access should raise if closed
        except Exception:
            failed = True
        assert failed


def test_lazytensor_h5_getitem_variants(fake_photon_library):
    # Exercise various __getitem__ forms against an h5py dataset source
    f = h5py.File(fake_photon_library, 'r', swmr=True, libver='latest')
    ds = f['vis']
    lt = LazyTensor(ds, dtype=torch.float32)

    n = len(lt)
    # scalar int
    assert torch.allclose(lt[0], torch.as_tensor(ds[0], dtype=torch.float32))

    # simple slice
    assert torch.allclose(lt[1:4], torch.as_tensor(ds[1:4], dtype=torch.float32))

    # tuple of slices (rows, cols)
    assert torch.allclose(
        lt[1:4, 2:7], torch.as_tensor(ds[1:4, 2:7], dtype=torch.float32)
    )

    # list of row indices (non-monotonic)
    rows = [5, 2, 7]
    lt_rows = lt[rows]
    ds_rows = torch.stack([torch.as_tensor(ds[i], dtype=torch.float32) for i in rows], dim=0)
    assert torch.allclose(lt_rows, ds_rows)

    # numpy array of row indices (non-monotonic)
    np_idx = np.array([3, 1, 6], dtype=np.int64)
    lt_rows = lt[np_idx]
    ds_rows = torch.stack([torch.as_tensor(ds[i], dtype=torch.float32) for i in np_idx.tolist()], dim=0)
    assert torch.allclose(lt_rows, ds_rows)

    # torch tensor of row indices
    t_idx = torch.tensor([4, 0, 2], dtype=torch.long)
    lt_rows = lt[t_idx]
    ds_rows = torch.stack([torch.as_tensor(ds[i], dtype=torch.float32) for i in t_idx.tolist()], dim=0)
    assert torch.allclose(lt_rows, ds_rows)

    # boolean mask (numpy)
    mask_np = np.zeros(n, dtype=bool)
    mask_np[[0, 2, n - 1]] = True
    lt_rows = lt[mask_np]
    ds_rows = torch.stack([torch.as_tensor(ds[i], dtype=torch.float32) for i in np.where(mask_np)[0].tolist()], dim=0)
    assert torch.allclose(lt_rows, ds_rows)

    # boolean mask (torch)
    mask_t = torch.zeros(n, dtype=torch.bool)
    mask_t[[1, 3, n - 2]] = True
    lt_rows = lt[mask_t]
    ds_rows = torch.stack([torch.as_tensor(ds[i], dtype=torch.float32) for i in torch.nonzero(mask_t, as_tuple=False).flatten().tolist()], dim=0)
    assert torch.allclose(lt_rows, ds_rows)

    # cleanup
    del lt
    f.close()

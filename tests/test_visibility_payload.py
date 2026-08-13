import h5py
import numpy as np
import pytest
import torch

from photonlib import PhotonLib, VoxelMeta


@pytest.fixture
def compact_meta():
    return VoxelMeta(
        shape=(2, 2, 2),
        ranges=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
    )


def write_library(path, meta, visibility):
    with h5py.File(path, "w") as output:
        output.create_dataset("numvox", data=meta.shape.cpu().numpy())
        output.create_dataset("min", data=meta.ranges[:, 0].cpu().numpy())
        output.create_dataset("max", data=meta.ranges[:, 1].cpu().numpy())
        output.create_dataset("vis", data=visibility.cpu().numpy())


@pytest.mark.parametrize("payload_shape", [(4,), (4, 3), (2, 3, 4)])
@pytest.mark.parametrize("lazy", [False, True])
def test_visibility_preserves_payload_dimensions(
    compact_meta, payload_shape, lazy, tmp_path
):
    visibility = torch.arange(
        len(compact_meta) * np.prod(payload_shape), dtype=torch.float32
    ).reshape(len(compact_meta), *payload_shape)

    if lazy:
        path = tmp_path / "payload.h5"
        write_library(path, compact_meta, visibility)
        library = PhotonLib.load(str(path), lazy=True)
    else:
        library = PhotonLib(compact_meta, visibility)

    inside = compact_meta.voxel_to_coord(torch.tensor([0, 5]))
    outside = torch.tensor([[3.0, 3.0, 3.0]])
    positions = torch.vstack((inside, outside))

    result = library.visibility(positions)

    assert result.shape == (3, *payload_shape)
    assert torch.equal(result[0], visibility[0])
    assert torch.equal(result[1], visibility[5])
    assert torch.count_nonzero(result[2]) == 0


@pytest.mark.parametrize("payload_shape", [(4,), (4, 3)])
@pytest.mark.parametrize("lazy", [False, True])
def test_visibility_scalar_queries_preserve_payload_dimensions(
    compact_meta, payload_shape, lazy, tmp_path
):
    visibility = torch.arange(
        len(compact_meta) * np.prod(payload_shape), dtype=torch.float32
    ).reshape(len(compact_meta), *payload_shape)

    if lazy:
        path = tmp_path / "scalar-payload.h5"
        write_library(path, compact_meta, visibility)
        library = PhotonLib.load(str(path), lazy=True)
    else:
        library = PhotonLib(compact_meta, visibility)

    inside = compact_meta.voxel_to_coord(torch.tensor(3))
    outside = torch.tensor([3.0, 3.0, 3.0])

    assert library.visibility(inside).shape == payload_shape
    assert torch.equal(library.visibility(inside), visibility[3])
    assert library.visibility(outside).shape == payload_shape
    assert torch.count_nonzero(library.visibility(outside)) == 0

import h5py
import torch
import numpy as np


class LazyTensor:
    """LazyTensor is a wrapper around a tensor or an h5py dataset.

    It acts as a torch.Tensor but only loads the data when needed, and can be converted to a dense tensor when needed.
    Helpful for very large datasets that don't fit in memory.

    Parameters
    ----------
    source : torch.Tensor | h5py.File
        The source tensor or h5py file.
    dtype : torch.dtype, optional
        The dtype of the tensor. Default is torch.float32.
    device : torch.device, optional
        The device of the tensor. Default is torch.device('cpu').
    """
    def __init__(self, source, dtype:torch.dtype=torch.float32, device=None):
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else torch.device('cpu')

        if isinstance(source, (h5py.Dataset,)):
            self._src_type = 'h5py'
            self._h5_file = source.file
            self._h5_dataset = source
            self._length = source.shape[0]
        else:
            # convert list-like to tensor
            self._src_type = 'tensor'
            self._tensor = torch.as_tensor(source, dtype=self._dtype)
            if self._device.type != 'cpu':
                self._tensor = self._tensor.to(self._device)
            self._length = self._tensor.shape[0]

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._get_source().shape

    def __len__(self):
        return self._get_source().shape[0]

    def to(self, device=None):
        if device is None or torch.device(device) == self._device:
            return self
        return LazyTensor(self._get_source(), self._dtype, device)

    def materialize(self):
        """load the entire data to a dense tensor on device. no-op if is a tensor"""
        if self._src_type == 'h5py':
            arr = self._h5_dataset[:]
            tens = torch.as_tensor(arr, dtype=self._dtype)
            if self._device.type != 'cpu':
                tens = tens.to(self._device)
            return tens
        else:
            return self._tensor

    def _get_source(self):
        if self._src_type == 'h5py':
            return self._h5_dataset
        return self._tensor

    def __getitem__(self, index):
        if self._src_type == 'h5py':
            arr = self._h5_dataset[index]
            tens = torch.as_tensor(arr, dtype=self._dtype)
            if self._device.type != 'cpu':
                tens = tens.to(self._device)
            return tens
        else:
            return self._tensor[index]

    def __del__(self):
        if self._src_type == 'h5py':
            self._h5_file.close()
import h5py
import torch
import numpy as np


class _H5Resource:
    """shared resource for an h5py file+dataset with simple ref counting"""
    def __init__(self, h5_file, dataset):
        self.file = h5_file
        self.dataset = dataset
        self.refcount = 1

    def incref(self):
        self.refcount += 1

    def decref(self):
        self.refcount -= 1
        if self.refcount <= 0:
            try:
                if self.file is not None:
                    self.file.close()
            except Exception:
                pass
            self.file = None
            self.dataset = None


class LazyTensor:
    """LazyTensor is a wrapper around a tensor or an h5py dataset.

    It acts as a torch.Tensor but only loads the data when needed, and can be converted to a dense tensor when needed.
    Helpful for very large datasets that don't fit in memory.

    Parameters
    ----------
    source : torch.Tensor | h5py.Dataset
        The source tensor or h5py dataset (e.g. h5py.File['vis']).
    dtype : torch.dtype, optional
        The dtype of the tensor. Default is torch.float32.
    device : torch.device, optional
        The device of the tensor. Default is torch.device('cpu').
    """
    def __init__(self, source, dtype:torch.dtype=torch.float32, device=None):
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else torch.device('cpu')

        if isinstance(source, LazyTensor): # for `.to()`
            self._src_type = source._src_type
            if self._src_type == 'h5py':
                self._h5 = source._h5
                if self._h5 is not None:
                    self._h5.incref()
                self._length = source._length
            else:
                tens = source._tensor
                if tens.dtype != self._dtype:
                    tens = tens.to(self._dtype)
                if self._device.type != 'cpu':
                    tens = tens.to(self._device)
                self._tensor = tens
                self._length = self._tensor.shape[0]
        elif hasattr(source, 'file') and hasattr(source, 'id'):
            self._src_type = 'h5py'
            self._h5 = _H5Resource(source.file, source)
            self._length = source.shape[0]
        else:
            # convert list-like to tensor
            self._src_type = 'tensor'
            self._tensor = torch.as_tensor(source, dtype=self._dtype)

            # Handle 0-dim tensors
            if self._tensor.dim() == 0:
                self._tensor = self._tensor.unsqueeze(0)

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
        # keep shared ownership of h5 resource when copying
        return LazyTensor(self, self._dtype, device)

    def materialize(self):
        """load the entire data to a dense tensor on device. no-op if is a tensor"""
        if self._src_type == 'h5py':
            arr = self._h5.dataset[:]
            tens = torch.as_tensor(arr, dtype=self._dtype)
            if self._device.type != 'cpu':
                tens = tens.to(self._device)
            return tens
        else:
            return self._tensor

    def _get_source(self):
        if self._src_type == 'h5py':
            return self._h5.dataset
        return self._tensor

    def __getitem__(self, index):
        if self._src_type == 'h5py':
            # h5py only supports advanced (fancy) indexing when indices are strictly increasing.
            # so if random order is needed, we need to assemble them manually.
            ds = self._h5.dataset

            # scalar/slice indexing
            if isinstance(index, (int, np.integer)) or isinstance(index, slice) or (
                isinstance(index, tuple) and all(isinstance(i, slice) for i in index)
            ):
                arr = ds[index].astype(np.float32)
                tens = torch.as_tensor(arr)
                if self._device.type != 'cpu':
                    tens = tens.to(self._device)
                return tens

            # 1D index (list, tuple, or torch.Tensor)
            if isinstance(index, torch.Tensor):
                if index.dtype == torch.bool:
                    idx_list = torch.nonzero(index, as_tuple=False).flatten().cpu().tolist()
                else:
                    idx_list = index.flatten().cpu().tolist()
            elif isinstance(index, np.ndarray):
                if index.dtype == np.bool_:
                    idx_list = np.nonzero(index)[0].tolist()
                else:
                    idx_list = index.flatten().tolist()
            elif isinstance(index, (list, tuple)):
                idx_list = list(index)
            else:
                arr = ds[index].astype(np.float32)
                tens = torch.as_tensor(arr, dtype=self._dtype)
                if self._device.type != 'cpu':
                    tens = tens.to(self._device)
                return tens

            rows = [ds[i] for i in idx_list]
            arr = np.stack(rows, axis=0).astype(np.float32)
            tens = torch.as_tensor(arr, dtype=self._dtype)
            if self._device.type != 'cpu':
                tens = tens.to(self._device)
            return tens
        else:
            return self._tensor[index]

    def __del__(self):
        if hasattr(self, '_src_type') and self._src_type == 'h5py':
            try:
                if hasattr(self, '_h5') and self._h5 is not None:
                    self._h5.decref()
            except Exception:
                pass

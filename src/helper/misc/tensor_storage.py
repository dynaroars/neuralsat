from beartype import beartype
import torch

class TensorStorage:
    
    """
    Fast managed dynamic sized tensor storage
    """
    
    @beartype
    def __init__(self: 'TensorStorage', full_shape: torch.Tensor, 
                 initial_size: int = 1024, switching_size: int = 65536, 
                 device: str = 'cpu', concat_dim: int = 0) -> None:
       
        if isinstance(full_shape, torch.Tensor):
            data = full_shape
            full_shape = data.shape
        else:
            data = None
            
        self.shape = list(full_shape)  
        self.dtype = torch.get_default_dtype()
        self.device = device
        self.concat_dim = concat_dim
        self.num_used = 0
        self.switching_size = switching_size
        
        self._storage = self._allocate(initial_size)

        if data is not None:
            self.append(data)


    @beartype
    def _allocate(self: 'TensorStorage', new_size: int) -> torch.Tensor:
        allocate_shape = self.shape.copy()
        allocate_shape[self.concat_dim] = new_size
        if self.device == 'cpu' and torch.cuda.is_available():
            # pin CPU memory if cuda is available
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device, pin_memory=True)
        else:
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device)


    @beartype
    def _get_new_size(self: 'TensorStorage', request_size: int) -> int:
        """Compute new size of storage given the current request."""
        if self._storage.size(self.concat_dim) < self.switching_size:
            # exponential growth with small tensor
            return max(self._storage.size(self.concat_dim) * 2, self.num_used + request_size)
        # linear growth with big tensor
        return self._storage.size(self.concat_dim) + request_size * 32


    @beartype
    @torch.no_grad()
    def append(self: 'TensorStorage', appended_tensor: torch.Tensor) -> 'TensorStorage':
        """Append a new tensor to the storage object."""
        if self.num_used + appended_tensor.size(self.concat_dim) > self._storage.size(self.concat_dim):
            # Reallocate a new tensor, copying the existing contents over.
            new_size = self._get_new_size(appended_tensor.size(self.concat_dim))
            new_tensor = self._allocate(new_size)
            new_tensor.narrow(dim=self.concat_dim, start=0, length=self.num_used).copy_(
                self._storage.narrow(dim=self.concat_dim, start=0, length=self.num_used))
            # And then remove the old storage object.
            del self._storage
            self._storage = new_tensor
        self._storage.narrow(self.concat_dim, self.num_used, appended_tensor.size(self.concat_dim)).copy_(appended_tensor)
        self.num_used += appended_tensor.size(self.concat_dim)
        return self
    
    
    @beartype
    @torch.no_grad()
    def pop(self: 'TensorStorage', size: int) -> torch.Tensor:
        """Remove tensors with 'size' at the end of the storage."""
        size = max(min(size, self.num_used), 0)
        ret = self._storage.narrow(self.concat_dim, self.num_used - size, size)
        self.num_used -= size
        return ret


    def __getattr__(self, attr):
        """Proxy all tensor attributes."""
        return getattr(self._storage.narrow(self.concat_dim, 0, self.num_used), attr)


    def __getitem__(self, idx):
        return self._storage.narrow(self.concat_dim, 0, self.num_used)[idx]


    def __len__(self):
        return self.num_used


    def __sub__(self, o):
        return self._storage.narrow(self.concat_dim, 0, self.num_used) - o._storage.narrow(o.concat_dim, 0, o.num_used)


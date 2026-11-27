import contextlib
import gc

import torch


def resolve_device(name='cpu', gpu_memory_fraction=None):
    """ Check a device string and, for a GPU, make it this process's current CUDA device.

        Parameters
        ----------
        name : str or torch.device, optional
            'cpu' (default) or an explicit GPU such as 'cuda:1'. A bare 'cuda' is
            rejected: on a shared machine it would silently mean GPU 0 for everyone.
            None is treated as 'cpu'.
        gpu_memory_fraction : float in (0, 1], optional
            Upper limit for PyTorch's caching allocator in this process, as a fraction
            of the GPU's TOTAL memory. This is a self-imposed limit, not an isolation
            mechanism: other processes are not restricted by it, and the CUDA context
            (a few hundred MB) is not counted. Ignored on CPU.

        Returns
        -------
        torch.device
    """
    if name is None:
        name = 'cpu'
    if isinstance(name, int):
        raise TypeError(f"Give the device as a string such as 'cuda:{name}' or 'cpu', not the integer {name!r}.")
    device = torch.device(name)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE is {str(name)!r}, but PyTorch sees no CUDA GPU here. Use 'cpu' or check the driver.")
    if device.index is None:
        raise ValueError("Give an explicit GPU, e.g. 'cuda:1' (or set CUDA_VISIBLE_DEVICES before "
                         "starting Python and use 'cuda:0'). A bare 'cuda' means GPU 0 on a shared machine.")
    n_gpus = torch.cuda.device_count()
    if device.index >= n_gpus:
        raise RuntimeError(f"DEVICE is {str(name)!r}, but only {n_gpus} GPU(s) are visible: cuda:0 to cuda:{n_gpus - 1}.")

    torch.cuda.set_device(device)  # so nothing lands on cuda:0 by default

    if gpu_memory_fraction is not None:
        if not 0 < gpu_memory_fraction <= 1:
            raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
        free, total = torch.cuda.mem_get_info(device)
        # Memory this process already holds in its cache can be reused, so count it as available.
        available = free + torch.cuda.memory_reserved(device)
        requested = gpu_memory_fraction * total
        if requested > available:
            raise RuntimeError(f"gpu_memory_fraction={gpu_memory_fraction} asks for {requested / 2**30:.1f} GiB "
                               f"on {device}, but only {available / 2**30:.1f} GiB is free. "
                               f"Lower the fraction or pick another GPU.")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
    return device


def release_gpu_memory(device):
    """ Return this process's cached, unused GPU memory to the driver, so other users can use it. """
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


@contextlib.contextmanager
def use_device(module, device='cpu', gpu_memory_fraction=None):
    """ Run a block with `module` on `device`, then move it back to the CPU.

        The model lives on the CPU between calls; the GPU is only occupied while
        the block runs, and its cached memory is released afterwards.

        Yields
        ------
        torch.device
            The resolved device.
    """
    device = resolve_device(device, gpu_memory_fraction)
    module.to(device)
    try:
        yield device
    finally:
        module.to("cpu")
        release_gpu_memory(device)


def rng_devices(device):
    """ CUDA device indices whose RNG state torch.random.fork_rng should save and restore. """
    device = torch.device(device)
    return [device.index] if device.type == "cuda" else []

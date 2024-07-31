import torch


def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    elif generator == "determ":
        return DeterministicGenerator(num_samples, seed)
    elif generator == "determ-indiv":
        return DeterministicIndividualGenerator(num_samples, seed)
    else:
        raise NotImplementedError


class DummyGenerator:
    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return torch.randn_like(*args, **kwargs)


class DeterministicGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a single rng and samples num_samples sized randomness and subsamples the current indices
    """

    def __init__(self, num_samples, seed=0):
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng = torch.Generator(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.set_seed(seed)

    def get_global_size_and_indices(self, size):
        global_size = (self.num_samples, *size[1:])
        indices = torch.arange(
            self.done_samples, 
            self.done_samples + int(size[0]),
        )
        indices = torch.clamp(indices, 0, self.num_samples - 1)
        return global_size, indices

    def get_generator(self):
        return self.rng

    def randn(self, *size, dtype=torch.float, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.randn(
            *global_size, 
            generator=generator, 
            dtype=dtype, 
            device=device
        )[indices]

    def randint(self, low, high, size, dtype=torch.long, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.randint(
            low, 
            high, 
            generator=generator, 
            size=global_size, 
            dtype=dtype, 
            device=device
        )[indices]

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        self.done_samples = done_samples
        self.set_seed(self.seed)

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.rng.manual_seed(seed)


class DeterministicIndividualGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a separate rng for each sample to reduce memoery usage
    """

    def __init__(self, num_samples, seed=0):
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng = [
            torch.Generator(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ) 
            for _ in range(num_samples)
        ]
        self.set_seed(seed)

    def get_size_and_indices(self, size):
        indices = torch.arange(
            self.done_samples,
            self.done_samples + int(size[0]),
        )
        indices = torch.clamp(indices, 0, self.num_samples - 1)
        return (1, *size[1:]), indices

    def get_generator(self, device):
        return self.rng

    def randn(self, *size, dtype=torch.float, device="cpu"):
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.cat(
            [
                torch.randn(
                    *size, 
                    generator=generator[i], 
                    dtype=dtype, 
                    device=device
                )
                for i in indices
            ],
            dim=0,
        )

    def randint(self, low, high, size, dtype=torch.long, device="cpu"):
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return torch.cat(
            [
                torch.randint(
                    low,
                    high,
                    generator=generator[i],
                    size=size,
                    dtype=dtype,
                    device=device,
                )
                for i in indices
            ],
            dim=0,
        )

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        self.done_samples = done_samples

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        [
            rng.manual_seed(i + self.num_samples * seed)
            for i, rng in enumerate(self.rng)
        ]

import os
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
):
    """Build torch dataloader from dataset
    args:
        dataset (BaseClassificationDataset): classification dataset
        batchSize (int): batch size
        numWorkers (int): the number of worker for loading dataset in parallel. Should be <= the number of CPU cores.
        shuffle (bool): whether to shuffle the dataset on the end of epoch
        pinMemory (bool): faster training with CUDA
    """

    # Fix pytorch only use 2 cores of CPU
    def worker_init_fn(worker_id):
        cpus = int(os.cpu_count())
        os.sched_setaffinity(0, list(range(cpus)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    return dataloader
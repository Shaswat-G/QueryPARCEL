from typing import List, Sequence

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader import DataLoader


class GNNDataloader(DataLoader):
    """Dataloader for torch_geometric dataset"""

    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str] | None = None,
        exclude_keys: List[str] | None = None,
        **kwargs,
    ):
        super().__init__(dataset, batch_size, shuffle, follow_batch, exclude_keys, **kwargs)

import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
from torch.utils.data import Dataset

from orbq.common.compat import loads as _loads
from orbq.common.typing import assert_is_instance as aii


class LmdbDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

        if isinstance(self.path, str):
            self.path = Path(self.path)

        self.env = None
        self.envs = None
        self._db_paths = []

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._db_paths = db_paths
            self._keys = []

            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    num_entries = cur_env.stat()["entries"]

                self._keys.append(list(range(num_entries)))
                cur_env.close()

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            cur_env = self.connect_db(self.path)
            num_entries = aii(cur_env.stat()["entries"], int)
            if cur_env.begin().get("length".encode("ascii")) is not None:
                num_entries -= 1
            self._keys = list(range(num_entries))
            self.num_samples = num_entries
            cur_env.close()

    def __len__(self):
        return self.num_samples

    def _init_db(self):
        if not self.path.is_file():
            self.envs = [self.connect_db(p) for p in self._db_paths]
        else:
            self.env = self.connect_db(self.path)

    def __getitem__(self, idx):
        if not self.path.is_file():
            if self.envs is None:
                self._init_db()

            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            datapoint_pickled = (
                self.envs[db_idx].begin().get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )

            data_object = _loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            if self.env is None:
                self._init_db()

            datapoint_pickled = self.env.begin().get(f"{self._keys[idx]}".encode("ascii"))
            data_object = _loads(datapoint_pickled)

        import torch

        for key, value in data_object:
            if isinstance(value, torch.Tensor) and value.requires_grad:
                setattr(data_object, key, value.detach())

        return data_object

    def get_metadata(self, num_samples):
        pass

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            if self.envs is not None:
                for env in self.envs:
                    env.close()
        else:
            if self.env is not None:
                self.env.close()
            self.env.close()

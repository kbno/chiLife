from .SpinLabel import SpinLabel
from functools import partial
import numpy as np
from tqdm import tqdm
# from joblib import Parallel, delayed
# import time


def _get_spin_label(frame, label, site, chain, protein, **kwargs):
    protein.universe.trajectory[frame]
    SL = SpinLabel(label, site=site, protein=protein, chain=chain, **kwargs)
    return SL


class SpinLabelTraj:
    def __init__(self, label=None, site=None, chain=None, protein=None, _label_traj=None, **kwargs):
        if _label_traj is not None:
            self.LabelTraj = list(_label_traj)
            return

        if protein is None:
            raise ValueError("`protein` must be provided when building a SpinLabelTraj from trajectory frames.")

        get_sl_frame = partial(
            _get_spin_label,
            label=label,
            site=site,
            chain=chain,
            protein=protein,
            **kwargs
        )

        n_frames = protein.universe.trajectory.n_frames
        self.LabelTraj = [
            get_sl_frame(i) for i in tqdm(np.arange(n_frames), desc="Making SpinLabelTraj")
        ]
        # self.LabelTraj = Parallel(n_jobs=n_jobs)(
        #     delayed(get_sl_frame)(i) for i in np.arange(n_frames)
        # )

    def __iter__(self):
        return iter(self.LabelTraj)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return self.LabelTraj[item]

        if isinstance(item, slice):
            return self.__class__(_label_traj=self.LabelTraj[item])

        if isinstance(item, (list, tuple, np.ndarray)):
            indices = np.asarray(item)

            if indices.dtype == bool:
                if indices.ndim != 1:
                    raise IndexError("Boolean mask must be one-dimensional.")
                if len(indices) != len(self.LabelTraj):
                    raise IndexError("Boolean mask must be the same length as SpinLabelTraj.")
                subset = [sl for sl, keep in zip(self.LabelTraj, indices) if keep]
                return self.__class__(_label_traj=subset)

            subset = [self.LabelTraj[i] for i in indices.tolist()]
            return self.__class__(_label_traj=subset)

        raise TypeError("Indices must be integers, slices, integer arrays, or boolean masks.")

    def __len__(self):
        return len(self.LabelTraj)

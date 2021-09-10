import os
import random

import numpy as np
import torch


from npe.networks.utils import normalize
from npe.anim.skeleton import Skeleton


def dataset_filename(name):
    return "data_" + name + ".npz"


def dataset_size(name):
    size_mb = os.path.getsize(name) / (1024 * 1024)
    return "{0:0.2f}".format(size_mb) + ' Mb'


class AnimDataset(torch.utils.data.Dataset):
    def __init__(self, files, folder="", save_stats=None, norm=['meanstd'],
                 start_index=0, end_index=None):

        super().__init__()

        # Indexes in clip
        self._start_index = start_index
        self._end_index = end_index

        self._clips = None
        self._filename_indices = np.array([], dtype=np.int32)
        self._filenames = []

        for f in files:
            abs_f = os.path.join(folder, dataset_filename(f))
            print('Loading dataset: ' + f + " (" + dataset_size(abs_f) + ")")

            data = np.load(abs_f)

            self.handle_data_dict(data)

        self._filenames = np.asarray(self._filenames).flatten()

        del f, files, data

        print("All datasets loaded.")

        self.skeleton = Skeleton(root_projection=True)

        self._clips = self._clips[..., slice(start_index, end_index)]
        self._pose_to_clip_index_offset = 1
        self._clips_length = self._clips.shape[1]

        stats = {}

        if 'meanstd' in norm:
            self.mean = self._clips.mean(axis=(0, 1), keepdims=True, dtype=float)
            self.std = self._clips.std(axis=(0, 1), keepdims=True, dtype=float)

            self._clips = normalize(self._clips, self.mean, self.std)

            stats['mean'], stats['std'] = self.mean, self.std

        if save_stats is not None:
            np.savez(save_stats, **stats)

        print("Dataloader ready (", np.prod(self._clips.shape[:-1]), "poses)")

    def handle_data_dict(self, data_dict):
        """
        What to do with the dict loaded from the npz. It is possible to overload this method 
        to extend/modify the loader's behavior
        """

        c = data_dict['clips'].astype(float)

        if self._clips is not None:
            self._clips = np.concatenate([self._clips, c], axis=0)
        else:
            self._clips = c

        self._filename_indices = np.concatenate([
            self._filename_indices,
            data_dict['file_indices'] + len(self._filenames)
        ])

        self._filenames.extend(data_dict['filenames'])

        del c
        self._clips = self._clips[..., slice(self._start_index, self._end_index)]

    def __len__(self) -> int:
        return len(self._clips)

    def _pose_index_to_clip_index(self, idx):
        """turn the index of a pose into the index of its source clip."""
        if type(idx) == slice:
            start = idx.start if idx.start is not None else 0
            step = idx.step if idx.step is not None else 1
            stop = idx.stop if idx.stop is not None else len(self._clips)-1
            idx = list(range(start, stop, step))
            idx = [i//self._pose_to_clip_index_offset for i in idx]

        elif isinstance(idx, (int, np.integer)):
            idx = idx//self._pose_to_clip_index_offset

        elif isinstance(idx, (list, np.ndarray)):
            idx = [i//self._pose_to_clip_index_offset for i in idx]
        return idx

    def get_filenames(self, idx):
        """return the filename of the clip at the specified index. 
        :param idx: slice, int-like, list-like
        """
        idx = self._pose_index_to_clip_index(idx)
        return self._filenames[self._filename_indices[idx]]

    def __getitem__(self, idx):
        c = self._clips[idx]
        sample = {
            'clip': c,
            'file': self.get_filenames(idx)
        }

        return sample


class PoseDataset(AnimDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_to_clip_index_offset, features = self._clips.shape[-2:]
        self._clips = self._clips.reshape((-1, features))

    @property
    def input_dim(self):
        return len(self._clips[-1])


class TargetInClipDataset(PoseDataset):
    def __init__(self, target_joints_names=['right_fingers', 'left_fingers'], **kwargs):
        super().__init__(**kwargs)

        self._targets_names = target_joints_names
        self.targets_indices = []
        for t in target_joints_names:
            self.targets_indices.extend(self.skeleton[t].vector_indices())

    def __getitem__(self, idx):
        # find a suitable pose target in the dataset
        # index of the pose in the clip
        clip_index = idx % self._clips_length

        # generate target index
        tindex = random.randint(0, self._clips_length-1) - clip_index

        sample = {
            'starting_pose': self._clips[idx],
            'target_pose': self._clips[idx + tindex],
            'targets': self._clips[idx + tindex][self.targets_indices],
            'targets_names': self._targets_names,
            'targets_indices':  self.targets_indices,
            'files_starting': self.get_filenames(idx),
            'files_matched': self.get_filenames(idx + tindex)
        }

        return sample

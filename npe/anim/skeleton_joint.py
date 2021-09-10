from __future__ import annotations
import numpy as np

from npe.math.vec3 import Vec3


class SkeletonJoint:
    def __init__(self, name: str, parent: SkeletonJoint = None, index=0, position=np.zeros(3), index_offset=0):
        self.parent = parent
        self.name = name
        self.index = index
        self.children = []
        self.is_limb_end = False

        self._index_offset = index_offset
        self._position = position

        if parent is not None:
            self.parent.children.append(self)
            self.is_limb_end = True

    @property
    def position(self):
        return self._position.view(Vec3)

    @position.setter
    def position(self, pos):
        pos = pos.squeeze()
        assert pos.ndim == 1 and pos.shape[0] == 3
        self._position = pos

    def parent_list(self):
        parents = []
        j = self
        while j.parent is not None:
            parents.append(j.parent)
            j = j.parent
        return parents

    def children_list(self):
        children = self.children.copy()
        for c in self.children:
            children.extend(c.children_list())

        return children

    def vector_indices(self):
        i = self.index * 3
        return [i, i+1, i+2]

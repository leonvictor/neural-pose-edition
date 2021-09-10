from typing import Any
from ..skeleton_joint import SkeletonJoint

import copy

class IKTarget:
    """Wrapper around SkeletonJoint with an added active field so we can disable targets on the fly"""
    _joint: SkeletonJoint
    active: bool

    def __init__(self, joint, active=True):
        # avoid infinite recursion when setting attributes
        super().__setattr__('_joint', copy.deepcopy(joint))
        super().__setattr__('active', active)
    
    def __setattr__(self, name: str, value: Any) -> None:
        self._joint.__setattr__(name, value)

    def __getattr__(self, attr):
        return self._joint.__getattribute__(attr)
    
    def __deepcopy__(self, memo):
        return IKTarget(self._joint, self.active)
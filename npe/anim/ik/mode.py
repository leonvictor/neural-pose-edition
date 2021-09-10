from enum import Enum


class IKMode(Enum):
    FABRIK = 0
    NEURAL = 1

    @classmethod
    def names(cls):
        return [mode.name for mode in cls]

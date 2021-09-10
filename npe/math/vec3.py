import numpy as np

class Vec3(np.ndarray):
    """Vec3 in xyz format"""
    # TODO: handle axis order mapping
    def __new__(cls, *args):
        if len(args) == 3: # If 3 values are given consider that they are x, y, z 
            arr = [args[0], args[1], args[2]]
        elif len(args) == 1:
            assert len(args[0]) == 1
            arr = args[0]
        else:
            raise NotImplementedError()
        return np.asarray(arr).view(cls)

    def __array_finalize__(self, obj):
        return

    @property
    def x(self):
        return self[0]

    @x.setter
    def set_x(self, x):
        self[0] = x

    @property
    def y(self):
        return self[1]

    @y.setter
    def set_y(self, y):
        self[1] = y
    
    @property
    def z(self):
        return self[2]

    @z.setter
    def set_z(self, z):
        self[2] = z

    def xyz(self):
        return self

    def xzy(self):
        return self[[0,2,1]]
        
    def distance_to(self, pos):
        return np.linalg.norm(self - pos, axis=0)
    
    def __str__(self):
        return "Vec3 (x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z) + ")"
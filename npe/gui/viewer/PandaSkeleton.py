import numpy as np
import panda3d
from direct.gui.OnscreenText import TextNode
from panda3d.core import Filename, LineSegs, NodePath, rad2Deg, NodePath, PandaNode, LQuaternion, CollisionNode
import inspect
from .geom_primitives import GeomPrimitives


class JointNode:
    def __init__(self, id, parent, name='joint_node'):
        self.id = id
        self.children = []
        self.parent_joint = parent
        if parent is not None:
            self.parent_joint.children.append(self)
        self.name = name
        self.root = render.attach_new_node('JointNode root')
        self.root.set_tag("selectable", "")
        self.root.set_python_tag("owner", self)

        self.sphere = GeomPrimitives.Sphere(6)
        self.sphere = self.sphere.instance_to(self.root)

        # Initialize limb if the joint has a parent
        self.cylinder = None
        if self.parent_joint:
            self.cylinder = self.root.attach_new_node("limb")
            GeomPrimitives.Cylinder(7).instance_to(self.cylinder)

    def __getattr__(self, name):
        return self.root.__getattribute__(name)

    def set_pos(self, *args):
        self.root.set_pos(*args)
        self.update_limb()
        for child in self.children:
            child.update_limb()

    def reparent_to(self, node):
        self.root.reparent_to(node)
        self.sphere_radius = node.scale * 6.0
        self.cylinder_radius = node.scale * 4.0
        self.root.set_scale(self.sphere_radius)
        return self

    def set_color(self, r, g, b, a=0):
        self.sphere.set_color(r, g, b, a)

    def get_color(self):
        return self.sphere.get_color()

    def __str__(self):
        return self.name

    def update_limb(self):
        if not self.parent_joint:
            return

        self.cylinder.set_pos(0, 0, 0)  # not useful ?
        length = (self.root.get_pos() - self.parent_joint.root.get_pos()).length()
        if self.cylinder_radius < 0:
            self.cylinder_radius = 0.1 * length
        self.cylinder.set_scale(self.root.parent, length, self.cylinder_radius, self.cylinder_radius)
        self.cylinder.look_at(self.parent_joint.root)
        q = LQuaternion()
        q.set_from_axis_angle(90.0, (0, 0, 1))
        self.cylinder.set_quat(q * self.cylinder.get_quat())

# from https://stackoverflow.com/questions/1697501/staticmethod-with-property


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class SkeletonStyle:

    def __init__(self, joint_color, limb_color):
        self.joint_color = joint_color
        self.limb_color = limb_color

    def __getitem__(self, key):
        if key == 0:
            return self.joint_color
        if key == 1:
            return self.limb_color
        else:
            raise IndexError

    @classproperty
    def WRONG(cls):
        return SkeletonStyle((0.8, 0.0, 0.3, 1.0), (0.7, 0.1, 0.6, 1.0))

    @classproperty
    def GOOD(cls):
        return SkeletonStyle((0.2, 0.7, 0.4, 1.0), (0.3, 0.8, 0.7, 1.0))

    @classproperty
    def TRANSPARENT(cls):
        return SkeletonStyle((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))

    @classproperty
    def NEUTRAL(cls):
        return SkeletonStyle((43/255, 43/255, 64/255, 1.0), (0.7, 0.7, 0.9, 1.0))


class PandaSkeleton(NodePath):
    """ class to draw a skeleton. Unit should be around an human in meter (between 1.50 and 2 units) """

    def __init__(self, viewer, n_joints, scale=1.0, id=-1, external_root_transform=False, names=None):
        self._node = PandaNode('root_node')
        super().__init__(self._node)
        self.reparent_to(render)

        # TODO: get rid of viewer (cyclic dependency)
        self.viewer = viewer
        self.origin = [0, 0, 0]
        self.scale = scale
        self.targets = []

        self.has_root_velocity = False
        self._external_root_transform = external_root_transform

        # Initialize trajectory display options
        self.trajectory_node = NodePath()
        self.trajectory_line = LineSegs()
        self.trajectory_line.set_color(240/255., 149/255., 29/255., 1.0)
        self.trajectory_line.set_thickness(2)

        self.enable_root_motion = False

        self.reset_root_transform()

        self.joints = []
        for i in range(n_joints):
            if self._external_root_transform and i == 1:
                parent_id = None
            else:
                parent_id = self.parent_id[i]

            parent = self.joints[parent_id] if parent_id is not None else None  # berk
            name = names[i] if names is not None else "joint_node"
            joint = JointNode(i, parent, name)
            joint.reparent_to(self)
            self.joints.append(joint)

        self.create_id_text_node()
        self.set_id(id)

        self.set_color_style()

    def create_id_text_node(self):
        self.id_txt = TextNode("id")
        self.id_txt.set_text("id")

        self.id_txt_node = self.attach_new_node(self.id_txt)
        self.id_txt_node.set_pos(0, 0, 0)
        self.id_txt_node.set_light_off(1)
        self.id_txt_node.set_scale(10.0)
        self.id_txt_node.set_p(-90)
        # TODO: Put on the floor

    def set_id(self, id: int):
        self.id = id
        self.id_txt.set_text(str(self.id))

    def set_display_anim_id(self, b: bool):
        if not b:
            self.id_txt_node.detach_node()
        else:
            self.id_txt_node.reparent_to(self)

    def reset_root_transform(self):
        """Reset transform for this skeleton (pos: origin, rotation: none, scale: 1)"""
        self.set_pos(*self.origin)
        self.set_hpr(0, 0, 0)
        self.set_scale(self.scale)

    @property
    def is_enabled(self):
        return self.has_parent()

    def enable(self):
        self.reparent_to(render)

    def disable(self):
        self.detach_node()

    def set_color(self, joint_color, limb_color):
        for joint in self.joints:
            joint.sphere.set_color(*joint_color)
            if joint.parent_joint:
                joint.cylinder.set_color(*limb_color)

    def set_color_style(self, style: SkeletonStyle = SkeletonStyle.NEUTRAL):
        self.set_color(*style)

    def set_pose(self, pose):
        print("You have to overide this method")

    def display_trajectory(self):
        print("You have to overide this method")

    def add_target(self, position=(0, 0, 0), scale=1, tag=None):
        cube = GeomPrimitives.Sphere(6)
        cube.reparent_to(self)
        cube.set_name(str(self) + " - Target (" + str(len(self.targets)) + ")")
        cube.set_pos(*position)
        cube.set_scale(scale)
        cube.set_tag('selectable', '')
        cube.set_color((1, 0, 0, 1))
        self.targets.append(cube)

    def get_targets(self, index=None):
        if index is not None:
            return self.targets[index]
        return self.targets

    def clear_targets(self):
        for t in self.targets:
            t.detach_node()
        self.targets = []

    def get_joint(self, name):
        # TODO: Better joint indexing
        for j in self.joints:
            if j.name == name:
                return j
        return None
# ==========================================================================================================
# ** Holden's format (in the NPZ files) : 73 values = 22 Positions(XYZ) + VX, VZ, VangY + 4 contacts
#  0 ** 0, 1, 2 : projection du root sur le sol ?       -1 (father)
#  1 ** 3, 4, 5 : root; vertebre basse                  -1
#  2 ** 6, 7, 8 : hanche gauche                         1
#  3 ** 9, 10, 11 : genou gauche                        2
#  4 ** 12, 13, 14 : cheville gauche                    3
#  5 ** 15, 16, 17 : orteils gauche                     4
#  6 ** 18, 19, 20 : hanche droite                      1
#  7 ** 21, 22, 23 : genou droite                       6
#  8 ** 24, 25, 26 : cheville droite                    7
#  9 ** 27, 28, 29 : orteils droite                     8
# 10 ** 30, 31, 32 : vertebre milieu                    1
# 11 ** 33, 34, 35 : vertebre haute                     10
# 12 ** 36, 37, 38 : nuque                              11
# 13 ** 39, 40, 41 : tete                               12
# 14 ** 42, 43, 44 : epaule gauche                      12
# 15 ** 45, 46, 47 : coude gauche                       14
# 16 ** 48, 49, 50 : poignet gauche                     15
# 17 ** 51, 52, 53 : doigts gauche                      16
# 18 ** 54, 55, 56 : epaule droite                      12
# 19 ** 57, 58, 59 : coude droite                       18
# 20 ** 60, 61, 62 : poignet droite                     19
# 21 ** 63, 64, 65 : doigts droite                      20
# xx ** 66 : Vitesse X
# xx ** 67 : Vitesse Z
# xx ** 68 : Vitesse angulaire autour de Y
# xx ** 69, 70 : Contact pied gauche
# xx ** 71, 72 : Contact pied droit
# ==========================================================================================================


class HPAPandaSkeleton(PandaSkeleton):
    def __init__(self, viewer=None, parents=None, n_joints=22, has_root_velocity=False, external_root_transform=False, names=None, ignore_first=False):
        if parents is not None:
            self.parent_id = parents
        else:
            self.parent_id = np.array([-1, -1, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20])

        super().__init__(viewer=viewer, n_joints=n_joints, scale=0.1, external_root_transform=external_root_transform, names=names)
        if ignore_first:
            self.joints[0].sphere.detach_node()
        self.has_root_velocity = has_root_velocity

    def initialize(self, data: np.ndarray):
        self.reset_root_transform()
        self.set_pose(data)

    def set_pose(self, data: np.ndarray):
        data = np.reshape(data, (-1, 3))
        for i, joint in enumerate(self.joints):
            # Panda3D uses different coordinate axis than HPA
            # TODO: specify which joint should be fixed at 0, 0, 0
            joint.set_pos(data[i, 0], data[i, 2], data[i, 1])

    def update(self, data: np.ndarray):
        self.set_pose(data)
        self.set_root_motion(data)

    def set_root_motion(self, data: np.ndarray, frame: int):
        if self.enable_root_motion and self.has_root_velocity and len(data.shape) > 1:
            f = data.shape[0]
            data = data.reshape((f, -1, 3))
            data = np.cumsum(data, axis=0) * self.scale
            self.set_hpr(rad2Deg(data[frame, -1, 2]), 0, 0)
            self.set_pos(self.origin[0] + data[frame, -1, 0], self.origin[1] + data[frame, -1, 1], 0)

    def display_trajectory(self, data: np.ndarray):
        # TODO: Do not update if the line is already drawn
        if not self.has_root_velocity:
            return

        self.hide_trajectory()

        f, _ = data.shape
        data = data.reshape((f, -1, 3))  # (frame, joints, xyz)
        data = np.cumsum(data, axis=0) * self.scale

        self.trajectory_line.move_to(*self.origin)
        for f in data:
            self.trajectory_line.draw_to(self.origin[0] + f[-1, 0], self.origin[1] + f[-1, 1], 0)

        self.trajectory_node = NodePath(self.trajectory_line.create())
        self.trajectory_node.reparent_to(render)
        self.trajectory_line.reset()

    def hide_trajectory(self):
        self.trajectory_node.detach_node()


class HPASkeletonFactory:
    # TODO: Clean up

    def __init__(self, parents=None, n_joints=22, has_root_velocity=False, external_root_transform=False, names=None, ignore_first=False):
        self.parents = parents
        self.n_joints = n_joints
        self.has_root_velocity = has_root_velocity
        self._external_root_transform = external_root_transform
        self._names = names
        self._ignore_first = ignore_first

    def __call__(self, viewer):
        return HPAPandaSkeleton(
            viewer,
            parents=self.parents,
            n_joints=self.n_joints,
            has_root_velocity=self.has_root_velocity,
            external_root_transform=self._external_root_transform,
            names=self._names,
            ignore_first=self._ignore_first
        )

    @property
    def skeleton_type(self):
        return HPAPandaSkeleton

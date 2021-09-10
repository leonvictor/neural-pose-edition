
from __future__ import annotations

from typing import Any
# import numpy.typing as npt
import math
import sys
import os
import time

import numpy as np
import panda3d

from .PandaViewer import PandaViewer
from .PandaSkeleton import HPAPandaSkeleton, PandaSkeleton, SkeletonStyle
from enum import Enum, Flag, auto
import imgui


class TrajectoryDisplay(Enum):
    NONE = 0
    CURRENT_FRAME = 1
    FULL_ANIM = 2

    def __add__(self, n: int):
        """Overload + operator to allow looping over states"""
        return TrajectoryDisplay((self.value + n) % 3)


class AutorunMode(Enum):
    NONE = 0  # Pause at the end of the anim
    ANIM = 1  # Automatically repeat the anim
    BATCH = 2  # Auto play the next batch


class AnimPandaViewer(PandaViewer):
    skeletons: Any(np.ndarray, None)

    def __init__(self, skeleton, animations: np.ndarray = None, nb_anim_simultaneous=25, max_skel_aligned=3, autorun=False, win_title="AnimPandaViewer", framerate=60, *kwargs):
        PandaViewer.__init__(self, win_title=win_title, *kwargs)

        self.skeleton_type = skeleton
        self.axis3D.detach_node()
        self.anim_id = 0
        self.frame_id = 0
        self._display_anim_id = True
        self.framerate = framerate
        self.time_since_last_frame = None  # Hacky
        self.anims = animations
        self.display_trajectory = TrajectoryDisplay.NONE
        self.enable_root_motion = False
        self.max_skel_aligned = max_skel_aligned
        self.autorun_mode = AutorunMode.NONE

        self.show_color_menu = False

        # Skeleton colors.
        self.skeleton_color, self.joints_color = SkeletonStyle.NEUTRAL

        self.target_color = (1, 1, 1)

        length = int(math.sqrt(nb_anim_simultaneous))
        # square of skeleton length*2.5  X  length*2.5
        self.set_display_board_dim(length, length, 2.5, 2.5)

        self.skeletons = None

        # Creates the self.skeletons
        self.set_batch_size(nb_anim_simultaneous)

        self.add_key("o", "Animation: previous batch of animations")
        self.add_key("p", "Animation: Next batch of animations")
        self.add_key("i", "Animation: toggle animation id on the ground")
        self.add_key("&", "Animation: decrease batch size to 1")
        self.add_key("-", "Animation: decrease batch size")
        self.add_key("=", "Animation: increase batch size")
        self.add_key('l', "Animation: previous frame")
        self.add_key('m', "Animation: next frame")
        self.add_key("t", "Toggle trajectory display")
        self.add_key("y", "Toogle root motion")

    def set_batch_size(self, nb_anim_simultaneous):
        if self.skeletons is not None:
            for i in self.skeletons:
                i.root_node.remove_node()

        self.skeletons = np.empty(nb_anim_simultaneous, dtype=PandaSkeleton)

        for i in range(self.batch_size):
            skel = self.skeleton_type(self)
            skel.set_id(i)
            self.skeletons[i] = skel
            self.set_animation_pos_xy_from_id(i)
        if self.anims is not None:
            self.styles = [SkeletonStyle.NEUTRAL for _ in self.anims]
            self.set_pose(reset=True)

    def set_data(self, data, reset_frame=False):
        # TODO: Setting the data in the correct format should be the application's responsability
        while(len(data.shape) < 3):
            data = data[None, :]
        self.anims = data

        # TODO: save the style as a SkeletonStyle
        self.styles = [SkeletonStyle(
            self.joints_color, self.skeleton_color)] * data.shape[0]
        if reset_frame:
            self.frame_id = 0
        self.set_pose(reset=True)

    def set_display_board_dim(self, dimx, dimy, paddingx=2.5, paddingy=2.5):
        self.dimx = dimx
        self.dimy = dimy
        self.paddingx = paddingx
        self.paddingy = paddingy

    def get_current_pose(self, skeleton_idx=0):
        assert 0 <= skeleton_idx < self.anims.shape[0]
        return self.anims[skeleton_idx, self.frame_id]

    @property
    def minx(self):
        return -self.paddingx * self.dimx * 0.5

    @property
    def miny(self):
        return -self.paddingy * self.dimy * 0.5

    def get_pos_xy(self, x, y):
        return [self.minx + (x+0.5) * self.paddingx, self.miny + (y+0.5) * self.paddingy, 0]

    def set_animation_pos_xy(self, anim_id, x, y):
        # x = i % l
        # y = int( i / l)
        self.skeletons[anim_id].origin = self.get_pos_xy(x, y)

    def set_animation_pos_xy_from_id(self, id):
        """Set the origin of an anim with regard to the batch size"""
        y = id % self.dimy
        x = int(id / self.dimx)
        self.set_animation_pos_xy(id, x, y)

    @property
    def batch_size(self):
        """ return the batch size """
        return self.skeletons.shape[0]

    def next_batch(self):
        """ extract the next batch from the dataset """
        self.anim_id = (self.anim_id + self.batch_size) % self.anims.shape[0]
        for i in range(self.batch_size):
            id = (self.anim_id+i) % self.anims.shape[0]
            self.skeletons[i].set_id(id)
        self.set_pose(reset=True)
        self.draw_trajectory()
        self.print_anim_info()

    def previous_batch(self):
        """ extract the previous batch from the dataset """
        self.anim_id = (self.anim_id - self.batch_size) % self.anims.shape[0]
        for i in range(self.batch_size):
            id = (self.anim_id+i) % self.anims.shape[0]
            self.skeletons[i].set_id(id)
        self.set_pose(reset=True)
        self.draw_trajectory()
        self.print_anim_info()

    def set_pose(self, pid=-1, reset=False):
        """Update the active batch of skeletons (IDs from pid to pid + batch_size) with data of the current frame.
        Args: pid = firts id of skeleton to update
              reset = reset root transform
        """
        if pid >= 0:
            self.frame_id = pid
        for i in range(self.batch_size):
            id = (self.anim_id+i) % self.anims.shape[0]
            if reset or self.frame_id == 0:
                self.skeletons[i].reset_root_transform()
            self.skeletons[i].enable_root_motion = self.enable_root_motion
            # animation 0, pose 0 => 73 dim
            self.skeletons[i].set_pose(self.anims[id][self.frame_id])
            self.skeletons[i].set_display_anim_id(self.display_anim_id)
            self.skeletons[i].set_color_style(self.styles[id])
            self.skeletons[i].set_root_motion(self.anims[id], self.frame_id)

    @property
    def display_anim_id(self):
        return self._display_anim_id

    @display_anim_id.setter
    def display_anim_id(self, value):
        for i in range(self.batch_size):
            self.skeletons[i].set_display_anim_id(value)
        self._display_anim_id = value

    def draw_trajectory(self):
        for i in range(self.batch_size):
            id = (self.anim_id+i) % self.anims.shape[0]
            if self.display_trajectory == TrajectoryDisplay.FULL_ANIM:
                self.skeletons[i].display_trajectory(self.anims[id])
            elif self.display_trajectory == TrajectoryDisplay.NONE:
                self.skeletons[i].hide_trajectory()
            elif self.display_trajectory == TrajectoryDisplay.CURRENT_FRAME:
                raise NotImplementedError()
                # TODO: Display only the next frame trajectory

    def key_handler(self, key):
        if key == "b":
            if self.frame_id > 0:
                self.frame_id -= 1
            else:
                self.frame_id = self.anims.shape[1]-1
            self.print_anim_info()
            self.set_pose()
        elif key == "o":
            self.previous_batch()
        elif key == "p":
            self.next_batch()
        elif key == "i":
            self.print_anim_info()
            self.display_anim_id = not self.display_anim_id
        elif key == "&":
            self.set_batch_size(1)
            self.next_batch()
            print("batch_size="+str(self.batch_size))
        elif key == "=":
            self.set_batch_size(self.batch_size+1)
            self.next_batch()
            print("batch_size="+str(self.batch_size))
        elif key == "-":
            if self.batch_size > 1:
                self.set_batch_size(self.batch_size-1)
                self.next_batch()
        elif key == 'l':
            self.frame_id -= 1
            if self.frame_id < 0:
                self.frame_id = self.anims.shape[1]-1
            self.set_pose()
        elif key == 'm':
            # next frame
            self.frame_id += 1
            if self.frame_id >= self.anims.shape[1]:
                self.frame_id = 0
            self.set_pose()
            print("batch_size="+str(self.batch_size))

        elif key == "t":
            # Display trajectory
            self.display_trajectory += 1
            self.draw_trajectory()
            print(self.display_trajectory)

        elif key == "y":
            self.enable_root_motion = not self.enable_root_motion
        else:
            super().key_handler(key)

    def print_anim_info(self):
        print("animation info: anim_id="+str(self.anim_id) + "/" +
              str(self.anims.shape[0]) + "  frame_id="+str(self.frame_id)+"/"+str(self.anims.shape[1]))

    def animate(self):
        t = time.perf_counter()

        # Wait to comply with desired framerate
        if self.time_since_last_frame and (t - self.time_since_last_frame) < (1/self.framerate):
            time.sleep((1 / self.framerate) - (t - self.time_since_last_frame))

        if (self.frame_id == self.anims.shape[1]-1 and self.autorun_mode is AutorunMode.BATCH):
            self.next_batch()

        self.frame_id += 1
        if self.frame_id >= self.anims.shape[1]:
            if self.autorun_mode is not AutorunMode.NONE:
                self.frame_id = 0
            else:
                self.frame_id -= 1

        self.set_pose()

        self.time_since_last_frame = time.perf_counter()

    def set_styles(self, styles):
        for i, style in enumerate(styles):
            self.skeletons[i].set_color_style(style)

    def set_style_by_indices(self, indices, style):
        self.styles[indices] = style

    def main_gui(self):
        super().main_gui()
        # Main menu bar
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu('Options', True):
                if imgui.begin_menu('Autorun', True):
                    res, _ = imgui.menu_item(
                        'Disable', None, self.autorun_mode == AutorunMode.NONE, True)
                    if res:
                        self.autorun_mode = AutorunMode.NONE
                    res, _ = imgui.menu_item(
                        'Batch', None, self.autorun_mode == AutorunMode.BATCH, True)
                    if res:
                        self.autorun_mode = AutorunMode.BATCH
                    res, _ = imgui.menu_item(
                        'Animation', None, self.autorun_mode == AutorunMode.ANIM, True)
                    if res:
                        self.autorun_mode = AutorunMode.ANIM

                    imgui.end_menu()

                if imgui.begin_menu('Ground display', True):
                    res, state = imgui.menu_item(
                        'Plane', None, self.ground.plane_display, True)
                    if res:
                        self.ground.show_plane(state)
                    res, state = imgui.menu_item(
                        'Grid', None, self.ground.grid_display, True)
                    if res:
                        self.ground.show_grid(state)
                    imgui.end_menu()

                clicked, state = imgui.menu_item('Color Scheme', None, False, True)
                if clicked:
                    self.show_color_menu = not self.show_color_menu

                imgui.end_menu()

            # Select other stuff
            imgui.end_main_menu_bar()

        # Color options
        if self.show_color_menu:
            imgui.begin("Colors options")
            mod, self.background_color = imgui.color_edit3(
                "Background", *self.background_color)
            if mod:
                base.set_background_color(self.background_color)
            mod_s, self.skeleton_color = imgui.color_edit4(
                "Skeleton", *self.skeleton_color)
            mod_j, self.joints_color = imgui.color_edit4(
                "Joints", *self.joints_color)
            if mod_j or mod_s:
                self.styles = [SkeletonStyle(
                    self.joints_color, self.skeleton_color)] * len(self.skeletons)
                for skel in self.skeletons:
                    skel.set_color(self.joints_color, self.skeleton_color)

            mod_t, self.target_color = imgui.color_edit3(
                "Targets", *self.target_color)
            if mod_t:
                # turn all targets in this color
                for skel in self.skeletons:
                    for t in skel.targets:
                        t.set_color(*self.target_color)
            mod, self.selected_color = imgui.color_edit3(
                "Targets (selected)", *self.selected_color)
            if (mod or mod_t) and self.selected_object is not None:
                self.selected_object.set_color(*self.selected_color)

            imgui.end()


if __name__ == '__main__' and sys.flags.interactive == 0:
    data = np.load(
        "D:\\Users\\leonv\\Dev\\LIRIS\\holden\\motionsynth_data\\data\\processed\\data_cmu_1.npz")
    # data = np.load('/d/Users/leonv/Dev/LIRIS/holden/motionsynth_data/data/processed/data_cmu_1.npz')
    # data = np.load( dataset.dir_anims_processed_npz + '/data_styletransfer.npz')
    anims = data['clips']

    #database = np.load( dataset.name2filename("styletransfer") )['clips']
    #database = np.load( dataset.name2filename("emilya") )['clips']
    #database = np.load( dataset.name2filename("cmu") )['clips']

    app = AnimPandaViewer(HPAPandaSkeleton, anims)
    app.run()

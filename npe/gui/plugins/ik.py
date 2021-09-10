from npe.gui.viewer.PandaViewer import GUI
import imgui
from npe.anim.ik.mode import IKMode


class IKGUI(GUI):
    def __init__(self, n_skeletons=1, default_modes=[]):
        super().__init__()

        if len(default_modes) > 0:
            assert(len(default_modes) == n_skeletons)
            self.ik_modes = default_modes
        else:
            self.ik_modes = [IKMode.FABRIK]*n_skeletons

        self._available_modes = IKMode.names()
        self._n_skeletons = n_skeletons
        self._selected_skeleton = 0

    def display(self):
        # IK options
        imgui.begin("IK Modes")
        for skeleton_index in range(self._n_skeletons):
            mode_idx = self._available_modes.index(
                self.ik_modes[skeleton_index].name)
            _, mode_idx = imgui.combo(
                str(skeleton_index), mode_idx, self._available_modes)
            self.ik_modes[skeleton_index] = IKMode[self._available_modes[mode_idx]]

        imgui.end()

    def get_mode(self, skeleton_index=None) -> IKMode:
        assert(skeleton_index < self._n_skeletons)

        if skeleton_index is None:
            return self.ik_modes

        return self.ik_modes[skeleton_index]

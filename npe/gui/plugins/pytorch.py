import torch
from npe.gui.viewer.PandaViewer import GUI
import imgui


class TorchGUI(GUI):
    """
    pyanim plugin to manage pytorch options (device, optimizers...)
    """

    def __init__(self, device='cpu'):
        self.enabled = True
        self.reset = False
        self.display_steps = False
        self.continuous = True

        self._devices_list = ["cpu"]

        if torch.cuda.is_available():
            self._devices_list.append("cuda")

        self._device = self._devices_list.index(device)
        self.device_changed = False

    def display(self):
        imgui.begin("EXT IK")
        self.device_changed, self._device = imgui.combo(
            "Device", self._device, self._devices_list)

        _, self.continuous = imgui.checkbox('Continuous', self.continuous)
        _, self.enabled = imgui.checkbox('Enable', self.enabled)

        self.reset = imgui.button('Reset')
        imgui.end()

    @property
    def device(self):
        if self._devices_list[self._device] == "cuda":
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

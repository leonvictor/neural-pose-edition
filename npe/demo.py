from collections import OrderedDict
import copy

import torch
import numpy as np

from npe.networks.ae.model import PoseAutoEncoder
from npe.networks.constraints.model import IKModel
from npe.networks.utils import load_config

from npe.anim.skeleton import Skeleton
from npe.anim.ik.target import IKTarget
from npe.anim.ik.mode import IKMode

from npe.gui.viewer.AnimPandaViewer import AnimPandaViewer
from npe.gui.viewer.PandaSkeleton import HPASkeletonFactory

from npe.gui.plugins import TorchGUI, StatsComparisonGUI, IKGUI

from npe.anim.ik.neural.combined_sequential import CombinedSequentialIKSolver


def load_stats(file: str):
    ms = np.load(file)
    return ms['mean'], ms['std']


class IKComparisonApp:
    def __init__(self, n_skeletons=2):
        self.n_skeletons = n_skeletons

        self._default_ik_modes = [IKMode.FABRIK, IKMode.NEURAL]

        self.config = load_config("./demo_config.yaml")
        self.mean, self.std = load_stats("stats.npz")
        self.skeleton = Skeleton(root_projection=True)

        device = torch.device("cpu")

        # Load base autoencoder
        self.ae = PoseAutoEncoder.from_weights(self.config['ae_weights']).to(device)

        # Initialize sequential solver
        self.skeleton.register_ik_solver(
            IKMode.NEURAL,
            CombinedSequentialIKSolver(
                self.ae.encode, self.ae.decode, self.skeleton,
                solvers=[IKModel.from_weights(w) for w in self.config['constraints_weights']],
                stats=(self.mean, self.std)
            ).to(device)
        )

        # Setup animation viewer
        self.viewer = AnimPandaViewer(
            HPASkeletonFactory(
                parents=self.skeleton.parents_indices,
                n_joints=self.skeleton.n_joints,
                has_root_velocity=False,
                external_root_transform=True,
                ignore_first=True
            ),
            nb_anim_simultaneous=self.n_skeletons,
            max_skel_aligned=self.n_skeletons
        )

        self.pytorch_gui = self.viewer.register_gui(TorchGUI(device=device.type))
        self.ik_gui = self.viewer.register_gui(IKGUI(n_skeletons, default_modes=self._default_ik_modes))
        self.stats_gui = self.viewer.register_gui(StatsComparisonGUI(IKMode.names()))

        # Load an initial pose
        initial_pose = np.load('demo_skeleton.npy')

        self.skeleton.set_pose(initial_pose)

        # Display the original animation
        self.initial_data = np.expand_dims(np.stack([initial_pose]*self.n_skeletons), 1)
        self.viewer.set_data(self.initial_data)
        self.viewer.animation_start()

        # Create default joints targets
        self.targets = OrderedDict()

        for name in self.skeleton.get_ik_solver(IKMode.NEURAL).targets:
            self.targets[name] = IKTarget(self.skeleton[name])

        self.initial_targets = copy.deepcopy(self.targets)
        self._set_targets_data(self.initial_targets)
        self.tol = 1e-1

        # Set default viewer options
        self.viewer.ground.enabled = True
        self.viewer.ground.show_plane(False)
        self.viewer.ground.grid_display = False
        self.viewer.display_anim_id = False

    def _sync_targets(self):
        """Check if a target moved on any skeleton, and synchronize the others if necessary"""
        for i, target in enumerate(list(self.targets.values())):
            for skel in self.viewer.skeletons:
                skel_target_pos = np.array(skel.get_targets(i).get_pos())

                # Update the target in dict if it moved
                if not np.allclose(skel_target_pos, target.position.xzy()):
                    target.position = skel_target_pos[[0, 2, 1]]
                    break

            # Update all the skeletons targets with the updated data
            for skel in self.viewer.skeletons:
                skel.get_targets(i).set_pos(*target.position.xzy())

    def _set_targets_data(self, new_targets):
        for skel in self.viewer.skeletons:
            skel.clear_targets()
            for target in list(new_targets.values()):
                skel.add_target(position=target.position.xzy(), scale=1)

    def run(self):
        while(self.viewer.step()):

            if self.pytorch_gui.device_changed:
                device = self.pytorch_gui.device
                self.ae.to(device)
                self.skeleton.get_ik_solver(IKMode.NEURAL).to(device)

            if self.pytorch_gui.reset:
                self.viewer.set_data(self.initial_data)
                self._set_targets_data(self.initial_targets)

            self._sync_targets()

            if self.pytorch_gui.enabled:
                if not self.pytorch_gui.continuous and self.viewer.is_mouse_active():
                    continue

                results = np.zeros((self.n_skeletons, 1, self.skeleton.dimensions))

                for skel_index in range(self.n_skeletons):
                    # First pose
                    self.skeleton.set_pose(self.viewer.get_current_pose(skel_index))

                    # Skip this step if the targets are already reached
                    if not self.skeleton.pose_close_to(self.targets, tol=self.tol):
                        mode = self.ik_gui.get_mode(skel_index)

                        self.stats_gui.start_timer(mode.name)

                        self.skeleton.solve_ik(self.targets, mode)

                        # Post process the pose
                        self.skeleton.fix_lengths()

                        self.stats_gui.stop_timer(mode.name)

                    results[skel_index] = self.skeleton.get_pose().ravel()

                self.viewer.set_data(results, reset_frame=False)


if __name__ == "__main__":
    app = IKComparisonApp(n_skeletons=2)
    app.run()

"""Animation data processing script.

Run this script to prepare multiple datasets from BVH to a unified format as proposed by Holden et al. 
Parameters:
 * target_framerate
 * window : number of frames each clip wil be composed of. Be careful : every original clip whose length is lower than window will be discarded
"""

from .Pivots import Pivots
from .Quaternions import Quaternions
from . import Animation
from . import BVH

import os
import sys
import numpy as np
import scipy.ndimage.filters as filters
from tqdm import tqdm
from enum import IntEnum, auto


class IndexEnum(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        return count


class OutputVectorIndices(IndexEnum):
    ROOT = auto()
    PELVIS = auto()
    LEFT_HIP = auto()
    LEFT_KNEE = auto()
    LEFT_FOOT = auto()
    LEFT_TOES = auto()
    RIGHT_HIP = auto()
    RIGHT_KNEE = auto()
    RIGHT_FOOT = auto()
    RIGHT_TOES = auto()
    SPINE_1 = auto()
    SPINE_2 = auto()
    NECK = auto()
    HEAD = auto()
    LEFT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    LEFT_WRIST = auto()
    LEFT_FINGERS = auto()
    RIGHT_SHOULDER = auto()
    RIGHT_ELBOW = auto()
    RIGHT_WRIST = auto()
    RIGHT_FINGERS = auto()


class BVHAxisOrder(IntEnum):
    z = 0
    y = 1
    x = 2


""""Emilya"""
# Emotions
emilya_actions_map = {
    "BS": "",
    "CS": "",
    "KD": "knocking_door",
    "WH": "walking_in_hand",
    "Th": "throwing",
    "W":  "walking",
    "SD": "sitting_down",
    "SDBS": "",
    "Lf": "lifting_object",
    "MB": "moving_book",
    "SW": "slow_walking",
}

emilya_emotion_map = {
    "Ag": "anger",
    "Sd": "sadness",
    "Ax": "anxiety",
    "Jy": "joy",
    "Nt": "neutral",
    "PF": "panic_fear",
    "Pr": "pride",
}

# Actions
""" hdm05 """
class_map = {
    'cartwheelLHandStart1Reps': 'cartwheel',
    'cartwheelLHandStart2Reps': 'cartwheel',
    'cartwheelRHandStart1Reps': 'cartwheel',
    'clap1Reps': 'clap',
    'clap5Reps': 'clap',
    'clapAboveHead1Reps': 'clap',
    'clapAboveHead5Reps': 'clap',
    # 'depositFloorR': 'deposit',
    # 'depositHighR': 'deposit',
    # 'depositLowR': 'deposit',
    # 'depositMiddleR': 'deposit',
    'depositFloorR': 'grab',
    'depositHighR': 'grab',
    'depositLowR': 'grab',
    'depositMiddleR': 'grab',
    'elbowToKnee1RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee1RepsRelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsRelbowStart': 'elbow_to_knee',
    'grabFloorR': 'grab',
    'grabHighR': 'grab',
    'grabLowR': 'grab',
    'grabMiddleR': 'grab',
    # 'hitRHandHead': 'hit',
    # 'hitRHandHead': 'grab',
    'hopBothLegs1hops': 'hop',
    'hopBothLegs2hops': 'hop',
    'hopBothLegs3hops': 'hop',
    'hopLLeg1hops': 'hop',
    'hopLLeg2hops': 'hop',
    'hopLLeg3hops': 'hop',
    'hopRLeg1hops': 'hop',
    'hopRLeg2hops': 'hop',
    'hopRLeg3hops': 'hop',
    'jogLeftCircle4StepsRstart': 'jog',
    'jogLeftCircle6StepsRstart': 'jog',
    'jogOnPlaceStartAir2StepsLStart': 'jog',
    'jogOnPlaceStartAir2StepsRStart': 'jog',
    'jogOnPlaceStartAir4StepsLStart': 'jog',
    'jogOnPlaceStartFloor2StepsRStart': 'jog',
    'jogOnPlaceStartFloor4StepsRStart': 'jog',
    'jogRightCircle4StepsLstart': 'jog',
    'jogRightCircle4StepsRstart': 'jog',
    'jogRightCircle6StepsLstart': 'jog',
    'jogRightCircle6StepsRstart': 'jog',
    'jumpDown': 'jump',
    'jumpingJack1Reps': 'jump',
    'jumpingJack3Reps': 'jump',
    'kickLFront1Reps': 'kick',
    'kickLFront2Reps': 'kick',
    'kickLSide1Reps': 'kick',
    'kickLSide2Reps': 'kick',
    'kickRFront1Reps': 'kick',
    'kickRFront2Reps': 'kick',
    'kickRSide1Reps': 'kick',
    'kickRSide2Reps': 'kick',
    'lieDownFloor': 'lie_down',
    'punchLFront1Reps': 'punch',
    'punchLFront2Reps': 'punch',
    'punchLSide1Reps': 'punch',
    'punchLSide2Reps': 'punch',
    'punchRFront1Reps': 'punch',
    'punchRFront2Reps': 'punch',
    'punchRSide1Reps': 'punch',
    'punchRSide2Reps': 'punch',
    'rotateArmsBothBackward1Reps': 'rotate_arms',
    'rotateArmsBothBackward3Reps': 'rotate_arms',
    'rotateArmsBothForward1Reps': 'rotate_arms',
    'rotateArmsBothForward3Reps': 'rotate_arms',
    'rotateArmsLBackward1Reps': 'rotate_arms',
    'rotateArmsLBackward3Reps': 'rotate_arms',
    'rotateArmsLForward1Reps': 'rotate_arms',
    'rotateArmsLForward3Reps': 'rotate_arms',
    'rotateArmsRBackward1Reps': 'rotate_arms',
    'rotateArmsRBackward3Reps': 'rotate_arms',
    'rotateArmsRForward1Reps': 'rotate_arms',
    'rotateArmsRForward3Reps': 'rotate_arms',
    # 'runOnPlaceStartAir2StepsLStart': 'run',
    # 'runOnPlaceStartAir2StepsRStart': 'run',
    # 'runOnPlaceStartAir4StepsLStart': 'run',
    # 'runOnPlaceStartFloor2StepsRStart': 'run',
    # 'runOnPlaceStartFloor4StepsRStart': 'run',
    'runOnPlaceStartAir2StepsLStart': 'jog',
    'runOnPlaceStartAir2StepsRStart': 'jog',
    'runOnPlaceStartAir4StepsLStart': 'jog',
    'runOnPlaceStartFloor2StepsRStart': 'jog',
    'runOnPlaceStartFloor4StepsRStart': 'jog',
    'shuffle2StepsLStart': 'shuffle',
    'shuffle2StepsRStart': 'shuffle',
    'shuffle4StepsLStart': 'shuffle',
    'shuffle4StepsRStart': 'shuffle',
    'sitDownChair': 'sit_down',
    'sitDownFloor': 'sit_down',
    'sitDownKneelTieShoes': 'sit_down',
    'sitDownTable': 'sit_down',
    'skier1RepsLstart': 'ski',
    'skier3RepsLstart': 'ski',
    'sneak2StepsLStart': 'sneak',
    'sneak2StepsRStart': 'sneak',
    'sneak4StepsLStart': 'sneak',
    'sneak4StepsRStart': 'sneak',
    'squat1Reps': 'squat',
    'squat3Reps': 'squat',
    'staircaseDown3Rstart': 'climb',
    'staircaseUp3Rstart': 'climb',
    'standUpKneelToStand': 'stand_up',
    'standUpLieFloor': 'stand_up',
    'standUpSitChair': 'stand_up',
    'standUpSitFloor': 'stand_up',
    'standUpSitTable': 'stand_up',
    'throwBasketball': 'throw',
    'throwFarR': 'throw',
    'throwSittingHighR': 'throw',
    'throwSittingLowR': 'throw',
    'throwStandingHighR': 'throw',
    'throwStandingLowR': 'throw',
    'turnLeft': 'turn',
    'turnRight': 'turn',
    'walk2StepsLstart': 'walk_forward',
    'walk2StepsRstart': 'walk_forward',
    'walk4StepsLstart': 'walk_forward',
    'walk4StepsRstart': 'walk_forward',
    'walkBackwards2StepsRstart': 'walk_backward',
    'walkBackwards4StepsRstart': 'walk_backward',
    'walkLeft2Steps': 'walk_left',
    'walkLeft3Steps': 'walk_left',
    'walkLeftCircle4StepsLstart': 'walk_left',
    'walkLeftCircle4StepsRstart': 'walk_left',
    'walkLeftCircle6StepsLstart': 'walk_left',
    'walkLeftCircle6StepsRstart': 'walk_left',
    'walkOnPlace2StepsLStart': 'walk_inplace',
    'walkOnPlace2StepsRStart': 'walk_inplace',
    'walkOnPlace4StepsLStart': 'walk_inplace',
    'walkOnPlace4StepsRStart': 'walk_inplace',
    'walkRightCircle4StepsLstart': 'walk_right',
    'walkRightCircle4StepsRstart': 'walk_right',
    'walkRightCircle6StepsLstart': 'walk_right',
    'walkRightCircle6StepsRstart': 'walk_right',
    'walkRightCrossFront2Steps': 'walk_right',
    'walkRightCrossFront3Steps': 'walk_right',
}

styletransfer_styles = [
    'angry', 'childlike', 'depressed', 'neutral',
    'old', 'proud', 'sexy', 'strutting']

styletransfer_motions = [
    'fast_punching', 'fast_walking', 'jumping',
    'kicking', 'normal_walking', 'punching',
    'running', 'transitions']

edin_locomotion_motion = ['walk', 'jog', 'run']

class_names = list(sorted(list(set(class_map.values()))))


def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def extract_class_from_filename(filename):
    """return style, action for this filename"""

    style = ""
    action = ""

    if 'emilya' in filename:
        cls_name = os.path.splitext(os.path.split(filename)[1])[0]
        style = ""
        action = ""
        for abbrev, name in emilya_emotion_map.items():
            if abbrev in cls_name:
                style = name
                break

    if filename.startswith('hdm05'):
        cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
        action = class_names.index(
            class_map[cls_name]) if cls_name in class_map else -1

    elif filename.startswith('edin_locomotion'):
        if "jog" in filename:
            action = "jog"
        if "run" in filename:
            action = "run"
        if "walk" in filename:
            action = "walk"

    return style, action


"""
Process one .bvh file:
 - Subsample to target_fps
 - Split in even clips

Args:
    * window [int]: Clip length
    * window_step [int]: Number of frames between each start of clips. If window_step < window the clips will overlap
    * target_fps [int]: 
    * border_mode [str]: Strategy to handle the clips that would be too short. 'drop' -> Don't 
    * grounding [str]: How to put anims on the floor. anim = the whole anim should be leveled to the ground
                                                      pose = each individual pose is grounded
    * reference [str]: ground|hip

"""


def process_file(
        filename: str,
        window: int = 240,
        window_step: int = 120,
        target_fps: int = 60,
        border_mode: str = 'drop',
        grounding: str = 'anim',
        reference: str = 'ground',
        keep_rotations: bool = False,
        foot_contact: bool = True,
        include_velocity: bool = True):

    assert grounding in ['anim', 'pose', '']
    assert border_mode in [
        'drop', 'repeat'], "Border mode must be one of ['drop', 'repeat']"
    assert reference in [
        'pelvis', 'root'], "Reference mode must be one of ['pelvis', 'root']"

    anim, names, frametime = BVH.load(filename)

    """ Convert to 60 fps """
    sampling = int(1/frametime) // target_fps
    anim = anim[::sampling]

    if len(anim) < window:
        # Skip shortest clips
        return None, None

    """ Do FK """
    global_positions = Animation.positions_global(anim)

    """ Remove Uneeded Joints """
    # Mapping from BVH skeleton to desired one
    indices = np.array([0,
                        2,  3,  4,  5,
                        7,  8,  9, 10,
                        12, 13, 15, 16,
                        18, 19, 20, 22,
                        25, 26, 27, 29])
    positions = global_positions[:, indices]

    if keep_rotations:
        global_rotations = Animation.rotations_global(anim)
        rotations = global_rotations[:, indices]

    """ Add Reference Joint (Projection of hip joint on the ground)"""
    trajectory_filterwidth = 3
    root_ref = positions[:, 0] * np.array([1, 0, 1])
    root_ref = filters.gaussian_filter1d(
        root_ref, trajectory_filterwidth, axis=0, mode='nearest')
    positions = np.concatenate([root_ref[:, np.newaxis], positions], axis=1)

    """ Handle vertical translation """
    left_foot_ids = np.array([OutputVectorIndices.LEFT_FOOT, OutputVectorIndices.LEFT_TOES])
    right_foot_ids = np.array([OutputVectorIndices.RIGHT_FOOT, OutputVectorIndices.RIGHT_TOES])

    # Feet should be the lowest point of the pose
    floor_y = np.minimum(
        positions[:, left_foot_ids, BVHAxisOrder.y],
        positions[:, right_foot_ids, BVHAxisOrder.y]
    ).min(axis=1)

    # The ground is found over the whole clip (to manage ladders etc)
    if grounding == 'anim':
        floor_y = softmin(floor_y, softness=0.5, axis=0)

    positions[..., BVHAxisOrder.y] -= np.expand_dims(floor_y, axis=1)

    """ Get Foot Contacts """
    if foot_contact:
        velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, left_foot_ids, BVHAxisOrder.x] - positions[:-1, left_foot_ids, BVHAxisOrder.x])**2
        feet_l_y = (positions[1:, left_foot_ids, BVHAxisOrder.y] - positions[:-1, left_foot_ids, BVHAxisOrder.y])**2
        feet_l_z = (positions[1:, left_foot_ids, BVHAxisOrder.z] - positions[:-1, left_foot_ids, BVHAxisOrder.z])**2
        feet_l_h = positions[:-1, left_foot_ids, BVHAxisOrder.y]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

        feet_r_x = (positions[1:, right_foot_ids, BVHAxisOrder.x] - positions[:-1, right_foot_ids, BVHAxisOrder.x])**2
        feet_r_y = (positions[1:, right_foot_ids, BVHAxisOrder.y] - positions[:-1, right_foot_ids, BVHAxisOrder.y])**2
        feet_r_z = (positions[1:, right_foot_ids, BVHAxisOrder.z] - positions[:-1, right_foot_ids, BVHAxisOrder.z])**2
        feet_r_h = positions[:-1, right_foot_ids, BVHAxisOrder.y]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

    """ Get Root Velocity """
    velocity = np.expand_dims(
        positions[1:, OutputVectorIndices.ROOT] - positions[:-1, OutputVectorIndices.ROOT],
        axis=1
    ).copy()

    """ Remove Translation """
    ref_index = OutputVectorIndices.PELVIS if reference == "pelvis" else OutputVectorIndices.ROOT
    positions[..., BVHAxisOrder.x] = positions[..., BVHAxisOrder.x] - \
        np.expand_dims(positions[:, ref_index, BVHAxisOrder.x], axis=1)
    positions[..., BVHAxisOrder.z] = positions[..., BVHAxisOrder.z] - \
        np.expand_dims(positions[:, ref_index, BVHAxisOrder.z], axis=1)

    """ Get Forward Direction """
    across1 = positions[:, OutputVectorIndices.LEFT_HIP] - positions[:, OutputVectorIndices.RIGHT_HIP]
    across0 = positions[:, OutputVectorIndices.LEFT_SHOULDER] - positions[:, OutputVectorIndices.RIGHT_SHOULDER]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity  # ?
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    if include_velocity:
        positions = positions[:-1]

    positions = positions.reshape(len(positions), -1)

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    if include_velocity:
        positions = np.concatenate([positions, velocity[..., BVHAxisOrder.z]], axis=-1)
        positions = np.concatenate([positions, velocity[..., BVHAxisOrder.x]], axis=-1)
        positions = np.concatenate([positions, rvelocity], axis=-1)

    if foot_contact:
        positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

    """ Slide over windows """
    windows = []
    windows_rot = []

    def padding(slice, window, zero_velocity=False):
        left = slice[:1].repeat((window-len(slice)) // 2 + (window-len(slice)) % 2, axis=0)
        right = slice[-1:].repeat((window-len(slice))//2, axis=0)

        if zero_velocity:
            left[:, -7:-4] = 0.0  # Zero velocity
            right[:, -7:-4] = 0.0

        return np.concatenate([left, slice, right], axis=0)

    for j in range(0, len(positions) - window//8, window_step):
        """ If slice too small pad out by repeating start and end poses """
        clip = positions[j:j+window]

        if keep_rotations:
            clip_rotations = rotations[j:j+window]

        if len(clip) < window:
            if border_mode == 'repeat':
                clip = padding(clip, window, zero_velocity=True)
                if keep_rotations:
                    clip_rotations = padding(
                        clip_rotations, window, zero_velocity=False)
            elif border_mode == 'drop':
                continue

        if len(clip) != window:
            raise Exception()

        windows.append(clip)

        if keep_rotations:
            windows_rot.append(clip_rotations)

    return windows, windows_rot


def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def process_db(db, save_as, window, window_step, target_fps, verbose=True, split_in=1):
    db = os.path.join(input_folder, db)
    files = get_files(db)

    data = {
        'clips': [],
        'rotations': [],
        'styles': [],
        'actions': [],
        'file_indices': [],
        'filenames': []  # how do we save each clip's source without repeating it x times ?
    }

    # TODO current_file and file_idx are the same
    current_file = 0
    file_idx = 0
    current_part = 1
    for item in tqdm(files, desc="Processing " + db + ":"):
        current_file += 1

        n_clips, rots = process_file(item, window, window_step, target_fps,
                                     grounding=grounding,
                                     reference=reference,
                                     keep_rotations=keep_rotation,
                                     foot_contact=foot_contact,
                                     include_velocity=include_velocity)

        style, action = extract_class_from_filename(item)

        if n_clips:
            data['clips'] += n_clips
            data['file_indices'].extend([file_idx] * len(n_clips))
            # print(len(data['clips']), len(data['file_indices']))
            data['filenames'].append(item)
            data['styles'].extend([style] * len(n_clips))
            data['actions'].extend([action] * len(n_clips))
            file_idx += 1

        if rots:
            data['rotations'] += rots

        if split_in > 1 and current_file > (len(files)+1)/4:
            name = save_as + '_' + str(current_part)
            np.savez_compressed(name, **data)

            current_part += 1
            current_file = 0  # Reset file counter
            file_idx = 0
            data = {
                'clips': [],
                'rotations': [],
                'styles': [],
                'actions': [],
                'file_indices': [],
                'filenames': []  # how do we save each clip's source without repeating it x times ?
            }

    assert(np.array(data['file_indices']).max() == file_idx-1)

    if split_in > 1:
        save_as += '_' + str(current_part)

    np.savez_compressed(save_as, **data)


# Databases names and number of parts in which we should split them
dbs = [
    # ('styletransfer', 1),
    # ('hdm05', 1),
    ('edin_locomotion', 1),
    ('cmu', 4),
    ('edin_terrain', 1),
    # ('mhad', 1),
    ('emilya', 1),
    ('edin_punching', 1),
    ('edin_misc', 1),
    # # ('edin_xsens', 1),
    # ('affective', 1),
    # ('MPI', 1)
    # ( 'edin_kinect',
]

# Parameters
input_folder = 'D:/Dev/data/anim/'
output_folder = 'D:/Dev/data/npe_pub/'
target_fps = 30
window = 30
window_step = 30
grounding = 'pose'
reference = 'root'
keep_rotation = False
foot_contact = False
include_velocity = False

os.makedirs(output_folder, exist_ok=True)

for db in dbs:
    process_db(db[0], os.path.join(output_folder, "data_" + db[0]),
               window, window_step, target_fps, split_in=db[1])

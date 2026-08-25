import math

import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class GripperConfig:
    """Franka Hand geometry, in the panda_hand frame (+z points out of the grip).

    TCP_OFFSET_Z is an external Franka constant (F_T_EE) that appears nowhere in
    the URDF; the other two are read back off the URDF/mesh by
    panda_with_tool_urdf._finger_span() and re-checked at every build.
    """

    TCP_OFFSET_Z = 0.1034    # panda_hand -> TCP (Franka F_T_EE)
    FINGER_MOUNT_Z = 0.0584  # panda_finger_joint1 origin
    FINGERTIP_Z = 0.1122     # finger.obj extent above the mount

    # hand_to_tool weld pitch: rotates the tool's link-1 axis (tool frame +z)
    # off the finger axis (panda_hand +z) so the tool extends outward,
    # perpendicular to the fingers -- parallel to the ground when the fingers
    # point straight down -- instead of in line with them. Azimuth (which
    # horizontal direction the tool points) isn't controlled here; it comes
    # for free from the arm's own wrist joint (panda_joint7), which rotates
    # panda_hand about its own z-axis.
    TOOL_MOUNT_PITCH = math.pi / 2

    # hand_to_tool weld roll: spins the tool 90 deg about its own long axis
    # (tool frame +z) before the pitch above reorients that axis outward.
    # Rotating about +z leaves the pointing direction unchanged and only
    # twists the tool's own x-z bend plane -- phi's deflection plane (see
    # tool_geometry's module docstring) -- relative to the fingers.
    TOOL_MOUNT_ROLL = math.pi / 2


class ArmConfig:
    """Franka Panda arm pose defaults.

    NEUTRAL_JOINT_VALUES is panda-gym's standard "ready" pose for the 7 arm
    joints (panda_joint1..7, URDF indices 0-6 -- stable under tool splicing
    since the tool is always appended after these). Used both as the RL reset
    pose (panda_with_tool.PandaWithTool.reset) and as a sane starting
    configuration for GUI demos, so it lives in one place rather than being
    redefined per caller.
    """

    NEUTRAL_JOINT_VALUES = (0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79)


class DesignPriorConfig:

    # For l1 and l2
    L_MIN = 0.15
    L_MAX = 0.5 # meters; sane band for a link on the franka panda arm

    # tau[2] is phi, the deflection of link 2 from straight: phi=0 is a straight
    # rod (the longest tool), phi -> +-pi folds link 2 back along link 1. The
    # interior elbow angle is theta = pi - |phi| (unsigned: +phi and -phi are
    # mirror-image tools and share the same elbow).
    #
    # The bound keeps link 2 from folding back down into the gripper. The tool
    # tip sits at TCP_OFFSET_Z + l1 + l2*cos(phi) in the panda_hand frame and
    # must stay above the finger mount. Worst case is the shortest handle with
    # the longest head (l1=L_MIN, l2=L_MAX), giving a critical angle of
    #   arccos((FINGER_MOUNT_Z - TCP_OFFSET_Z - L_MIN) / L_MAX) = 1.971 rad
    # rounded down to 1.9 rad (109 deg), which leaves 3.3 cm of clearance above
    # the mount in that corner. This also excludes the degenerate fold near
    # phi=pi, where link 2 would lie buried inside link 1's own box.
    #
    # Tip-z-above-mount is a deliberately conservative proxy: at phi=1.9 the tip
    # is also displaced 0.47 m in x, so it is nowhere near the fingers laterally.
    PHI_MAX = 1.9

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

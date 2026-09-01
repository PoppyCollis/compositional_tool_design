import math

import torch

import se2

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

    # (lower, upper) for panda_joint1..7, from assets/franka_panda/panda.urdf.
    # Duplicated here rather than parsed because config.py does no I/O at import
    # and assets/ is gitignored; tests/test_se2_control.py re-checks them against
    # the live URDF, the same guard _check_tcp_offset gives the gripper constants.
    #
    # These are passed to calculateInverseKinematics, which otherwise ignores
    # limits entirely and happily winds panda_joint7 past its +-2.9671 stop. That
    # is not a harmless overshoot: PyBullet's motors *do* enforce limits when
    # stepping, so the joint saturates, the hand lands somewhere other than the
    # commanded pose, and the tool tilts with nothing reporting an error.
    JOINT_LIMITS = (
        (-2.9671, 2.9671),
        (-1.8326, 1.8326),
        (-2.9671, 2.9671),
        (-3.1416, 0.0),
        (-2.9671, 2.9671),
        (-0.0873, 3.8223),
        (-2.9671, 2.9671),
    )


class SE2Config:
    """Constants for the flat, fixed-height SE(2) controller (see se2.py).

    WORKSPACE is the single source of truth for where the hand may go: the
    per-step clipper, the reset sampler and the observation normaliser all read
    this one box. Defining it twice is how the three quietly drift apart.
    """

    # Height of the tool's centreline above the table top, which panda-gym puts at
    # z=0 (PyBullet.create_table: "Top is z=0"). Low enough to catch a block, high
    # enough to clear the surface.
    TOOL_Z = 0.02

    # World height of the panda_hand origin. With the fingers pointing straight down
    # the hand's +z axis is world -z, so the tool -- welded TCP_OFFSET_Z out along
    # that axis -- sits exactly this far below the hand, for every tau. See se2.py's
    # module docstring for why the tip's height carries no tau dependence.
    HAND_Z = TOOL_Z + GripperConfig.TCP_OFFSET_Z

    # Measured by workspace_sweep.py with the robot base at the origin: the largest
    # rectangle in which the arm can hold the tool flat (tilt < 1 deg, height within
    # 2 mm) at every yaw, inset by SWEEP_MARGIN. A robot based elsewhere translates
    # it (se2.Box.translate), which PandaWithTool does at construction.
    # Measured 2026-08-25: 81x101 cells at 1 cm over 12 yaws, largest all-clean
    # rectangle x[0.350, 0.650] y[-0.410, 0.410], inset by SWEEP_MARGIN.
    WORKSPACE = se2.Box(x_min=0.380, x_max=0.620, y_min=-0.380, y_max=0.380)

    # Hand yaw is clipped to +-YAW_LIMIT, exactly as x and y are clipped to
    # WORKSPACE, and the sweep only accepts a cell if it is clean across this whole
    # range. Yaw is not free: the tool points along the hand's +x axis, so yaw near
    # +-pi aims it back at the robot's own base, which needs panda_joint7 wound into
    # the dead band beyond its +-2.9671 stop. Restricting the range instead of
    # accepting a tiny workspace -- workspace_sweep.py prints the trade for a
    # spread of candidates. At +-90 deg the clean rectangle is 0.24 x 0.76 m;
    # +-45 deg would buy 50% more area but halve the headings the tool can take,
    # which matters more to a hooking policy than a few cm of depth.
    YAW_LIMIT = math.pi / 2

    # Inset applied to the swept region, for tracking error now and for the real arm
    # not matching the model later. The sweep says where it just barely works.
    SWEEP_MARGIN = 0.03

    # Further inset for reset sampling, so an episode never starts already pressed
    # against a wall with half its action range dead.
    RESET_MARGIN = 0.05

    # Metres and radians per unit action. Deliberately a fifth of panda-gym's 0.05
    # m/step: panda-gym constrains only position, so its transient tracking error
    # costs nothing, whereas here a lagging arm is a *tilted* one. Interpolating in
    # joint space between two flat IK solutions does not stay flat, so an untrackable
    # step dips the tool mid-motion even though both endpoints are level. Measured at
    # 0.05 the tool dived 41 mm and leaned 2.8 deg while moving; at 0.01 the worst
    # transient is 0.8 mm and 0.11 deg. Still 0.25 m/s at 25 Hz, which crosses the
    # workspace in ~24 steps.
    #
    # YAW_SCALE is twice POS_SCALE so that a ~0.5 m tool's tip, on its lever arm,
    # moves about as far per step from turning as from translating.
    POS_SCALE = 0.01
    YAW_SCALE = 0.02
    VEL_SCALE = 0.5    # m/s (and rad/s) normaliser for the velocity terms in get_obs

    # How far the commanded target may get ahead of where the hand actually is,
    # in metres and radians -- roughly three steps' worth.
    #
    # The target is integrated internally rather than re-read from the arm each
    # step. Re-reading sounds safer, and it does stop the target running past the
    # workspace, but it also feeds the IK solver's residual back into its own input:
    # the solver lands a fraction of a millimetre off, that becomes next step's
    # starting point, and the error compounds. Measured at 300 steps of *zero*
    # action, the hand crept up to 25 cm. An integrated target is exactly unchanged
    # by a zero action.
    #
    # The clamp then restores what re-reading was for. Clipping to WORKSPACE already
    # stops the target escaping the table; this stops it escaping the *arm* when the
    # hand is blocked by an object rather than a boundary, so the moment the
    # obstruction clears the arm does not lunge for a target metres away.
    MAX_LAG = 0.03
    MAX_YAW_LAG = 0.06

    # calculateInverseKinematics iterates from the current joint state, so a single
    # call will not converge over a large jump. Only reset and the sweep need this;
    # per-step targets are one POS_SCALE away and converge in one call.
    IK_ROUNDS = 8


class TaskConfig:
    """Object and target geometry for the task encoding (ai_docs/task_encoding_g.md).

    SE2Config.WORKSPACE says where the *hand* may go. These say where an *object*
    may sit, which is a different question and the one `p_target` is drawn from. The
    regions themselves are computed by task_space.py and mapped by reach_sweep.py;
    only the scalars live here, along with SCENE_BOX, the frozen normalisation map
    shared by the observation and by initial_state.h.
    """

    # Max object radius, as a plan-view disk. The maps are all computed at this
    # value: a smaller object is strictly easier to reach, so a region measured at
    # the maximum is valid for every object the task samples.
    #
    # The tool's cross-section spans z in [TOOL_Z - H/2, TOOL_Z + H/2] = [0.01, 0.03],
    # so an object shorter than ~1 cm is passed over rather than touched. At 3 cm
    # radius the object is 3x the tool's own 2 cm cross-section, so "getting around
    # it" stays a real geometric constraint rather than a rounding error.
    R_OBJ = 0.03

    # Circumradius of the closed gripper's planar footprint at TOOL_Z, measured from
    # the panda_hand origin. Measured 2026-08-27 by reach_sweep.py --measure-gripper:
    # only the two fingers reach the tool plane (they span z=[0.0075, 0.0692], while
    # panda_hand itself bottoms out at z=0.0538), with a circumradius of 0.0429 m.
    # Rounded up, which errs towards calling a target hand-reachable and so towards
    # *under*-claiming the region that genuinely needs a tool.
    #
    # Used only to define the bare-arm band WORKSPACE (+) Disk(R_OBJ + GRIPPER_RADIUS):
    # the counterfactual a tool has to beat. A target inside it is solvable with no
    # tool at all, so it carries no design signal whatever the policy does.
    GRIPPER_RADIUS = 0.045

    # The one box every planar position in the observation is normalised against --
    # hand, tool tip, object and target alike -- and the support of p_target.
    #
    # Deliberately NOT SE2Config.WORKSPACE. The tool's whole purpose is to put the tip
    # outside the arm's reach, so normalising on the hand's own rectangle would push
    # long tools past +-1: exactly the designs the search exists to evaluate would land
    # in the network's extrapolation region, and (dV/dx1) is half the design gradient.
    # The same argument applies to the object, which in sweeping and pushing crosses
    # the reach boundary in both directions, so no sub-box contains its trajectory.
    #
    # Extents: measured 2026-09-01, the union of tool-reachable object centres over
    # 2000 ToolPrior designs at R_OBJ is x[0.150, 1.035], y[-0.800, 0.795]. This box
    # contains it with padding. In the robot base frame, like SE2Config.WORKSPACE;
    # PandaWithTool translates it by the base position.
    #
    # Uniform target sampling over it is 39.7% unreachable by every prior design,
    # 49.2% in the discriminating band (0 < coverage < 1) and 11.1% reachable by all
    # (measured 2026-09-01, 5 mm grid, 2000 prior designs at R_OBJ).
    # The unreachable third is intentional -- see ai_docs/task_encoding_g.md -- but it
    # means evaluation has to stratify by band or it mostly measures the mixing ratio.
    SCENE_BOX = se2.Box(x_min=0.0, x_max=1.10, y_min=-1.0, y_max=1.0)

    # The support of p_target. The same rectangle as SCENE_BOX, named separately
    # because they are the same for two different reasons and only one is likely to
    # move: SCENE_BOX must contain everything the network ever sees, TARGET_BOX is a
    # choice about which tasks to pose.
    TARGET_BOX = SCENE_BOX

    # Success tolerance rho_target. A field of g rather than an eval constant, so the
    # objective and the metric cannot drift apart (ai_docs/task_encoding_g.md).
    RHO_TARGET = 0.03

    # Fixed episode length; no early termination on success. Terminating early would
    # truncate the accumulating negative reward, making V jump discontinuously at the
    # rho boundary and putting success and failure returns on different scales. V is
    # read as an energy over designs, so that comparability matters more here than the
    # wall-clock an early exit would save.
    HORIZON = 100


class DesignPriorConfig:

    # For l1 and l2
    L_MIN = 0.1
    L_MAX = 0.2 # meters; sane band for a link on the franka panda arm

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

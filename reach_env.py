"""ReachEnv: the simulator side of the initial-state map ``h``.

``initial_state.h(tau, g, xi)`` computes analytically, in torch, the state a design
lands an episode in. This module produces the same vector out of PyBullet, and every
subsequent state of the episode besides. The design objective

    f(tau, g) = E_xi[ V(h(tau, g, xi), tau, g) ]

is only meaningful if the two agree, so ``tests/test_reach_env.py`` pins them against
each other rather than trusting that both happen to call ``Box.normalise_point`` on
``TaskConfig.SCENE_BOX``.

Deliberately a plain ``gymnasium.Env`` and not ``panda_gym.envs.core.RobotTaskEnv``,
for three independent reasons:

- ``RobotTaskEnv`` hardcodes a HER-style Dict observation space
  (``observation``/``achieved_goal``/``desired_goal``); the observation here is one
  flat 21-vector, which is what ``h`` returns.
- It requires a ``panda_gym.envs.core.Task`` ABC. This repo's ``task.Task`` is an
  unrelated NamedTuple with the same name -- importing both invites a silent clash.
- Its ``step`` terminates on success, which ``TaskConfig.HORIZON``'s comment forbids:
  truncating the accumulating negative reward makes ``V`` jump at the ``rho``
  boundary, and ``V`` is read as an energy across designs.

``tau`` is fixed for the env's life. The URDF is written and loaded once, in
``PandaWithTool.__init__``; design diversity for PPO comes from N parallel envs each
holding one design, not from redrawing tau at reset.
"""
import numpy as np
import gymnasium
from gymnasium import spaces
from panda_gym.pybullet import PyBullet

import initial_state
import se2
import task as task_mod
from config import TaskConfig
from panda_with_tool import PandaWithTool

# Ghost marker appearance. Both are visual-only bodies (ghost=True): a real object
# would be knocked away by any tool that reached it, punishing the policy for
# succeeding (see task.object_start).
OBJECT_RGBA = np.array([0.1, 0.3, 0.9, 0.6])
TARGET_RGBA = np.array([0.1, 0.8, 0.3, 0.4])

# The target disc is a flat token on the table top, not a solid of any depth.
TARGET_DISC_HEIGHT = 0.001


class ReachEnv(gymnasium.Env):
    """One tool design, one task type, driven in SE(2) against a ghost target.

    Args:
        tau: Design parameters ``(l1, l2, phi)``. Fixed for the env's lifetime.
        task_id (task.TaskType): Which task instance to draw at reset. Only REACH
            has a start distribution; the others raise from ``task.TASK_PARAMS``.
        render_mode (str): ``"rgb_array"`` (headless) or ``"human"`` (GUI window).
        base_position (np.ndarray, optional): Robot base position, as (x, y, z).
        n_substeps (int): Physics substeps per env step. panda-gym's default of 20
            at its 1/500 s timestep gives 25 Hz control, which is what
            ``SE2Config.POS_SCALE``'s 0.01 m/step was chosen against.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, tau, task_id=task_mod.TaskType.REACH, render_mode="rgb_array",
                 base_position=None, n_substeps=20):
        self.tau = tuple(float(v) for v in tau)
        self.task_id = task_mod.TaskType(task_id)
        self.render_mode = render_mode

        # One client per env, so a later SubprocVecEnv needs no changes here.
        self.sim = PyBullet(render_mode=render_mode, n_substeps=n_substeps)
        self.metadata = dict(self.metadata, render_fps=1.0 / self.sim.dt)

        self.robot = PandaWithTool(self.sim, self.tau, base_position=base_position)
        # Read the normalisation box off the robot rather than re-deriving it, so the
        # two blocks of the observation cannot be normalised against different boxes.
        self.scene = self.robot.scene
        self._build_scene()

        self.action_space = self.robot.action_space
        # Bounded rather than infinite so an out-of-range value is a detectable bug:
        # positions are normalised by SCENE_BOX and velocities by VEL_SCALE, so the
        # achievable range is roughly +-2. A test asserts containment over a rollout.
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(initial_state.OBS_DIM,), dtype=np.float32
        )

        self._task = None
        self._xi = None
        self._elapsed = 0

        # The two assertions plan.md could not make without a sim up. ROBOT_DIM is
        # declared in initial_state.py, which cannot import PandaWithTool without
        # pulling PyBullet into a pure-torch module -- so this is where it gets
        # checked.
        n_robot = len(self.robot.get_obs())
        if n_robot != initial_state.ROBOT_DIM:
            raise AssertionError(
                f"initial_state.ROBOT_DIM is {initial_state.ROBOT_DIM} but "
                f"PandaWithTool.get_obs returns {n_robot} values"
            )
        # Unseeded on purpose. gymnasium only reseeds when seed is not None, so
        # passing a literal here would leave every identically-constructed env
        # sharing one stream -- 64 SubprocVecEnv workers replaying the same episodes
        # until something explicitly seeded them. With None, self.np_random draws
        # from OS entropy per process, and an explicit reset(seed=...) still pins it.
        self.reset()
        n_obs = len(self._get_obs())
        if n_obs != initial_state.OBS_DIM:
            raise AssertionError(
                f"initial_state.OBS_DIM is {initial_state.OBS_DIM} but the env "
                f"assembles {n_obs} values"
            )

    # -- scene ------------------------------------------------------------------

    def _build_scene(self):
        """Create the ghost markers, once.

        No support surface: nothing about reaching is dynamic, and a table sized
        naively around a base-at-origin robot intersects the Panda's own base link.
        Sweeping and pushing will need one -- ``sim.create_table`` is a real
        collision body with friction kwargs, not scenery -- and this is where it
        goes. See plan.md.
        """
        with self.sim.no_rendering():
            # Centre at r_obj so the sphere rests on the table top at z=0, the plane
            # SE2Config.TOOL_Z is measured from. The tool's cross-section spans
            # z in [0.01, 0.03], so it strikes the object below its equator.
            self.sim.create_sphere(
                body_name="object",
                radius=TaskConfig.R_OBJ,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, TaskConfig.R_OBJ]),
                rgba_color=OBJECT_RGBA,
            )
            # Radius rho, so the success tolerance is the thing drawn rather than an
            # arbitrary marker size.
            self.sim.create_cylinder(
                body_name="target",
                radius=TaskConfig.RHO_TARGET,
                height=TARGET_DISC_HEIGHT,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, TARGET_DISC_HEIGHT / 2.0]),
                rgba_color=TARGET_RGBA,
            )

    def _place_markers(self, task, xi):
        """Move the ghost bodies to this episode's object start and target."""
        flat = np.array([0.0, 0.0, 0.0, 1.0])
        obj = np.asarray(xi.obj_xy, dtype=float)
        target = np.asarray(task.target, dtype=float)
        self.sim.set_base_pose(
            "object", np.array([obj[0], obj[1], TaskConfig.R_OBJ]), flat
        )
        self.sim.set_base_pose(
            "target", np.array([target[0], target[1], TARGET_DISC_HEIGHT / 2.0]), flat
        )

    # -- state ------------------------------------------------------------------

    def _tip_xy(self):
        """Tool-tip position in the table plane, in metres.

        The closed form, not ``robot.get_ee_position()``: ``PandaWithTool.get_obs``
        and ``initial_state.h`` both use ``se2.tip_from_hand``, and mixing FK in here
        would make the reward disagree with the observation by the IK residual.
        """
        return se2.tip_from_hand(self.robot.get_hand_se2(), self.tau)

    def _object_xy(self):
        """Object position in the table plane, in metres.

        Read back off the body rather than cached from ``xi``. For a ghost that never
        moves this is a no-op indirection, but it is already the code path sweeping
        and pushing need, so no branch gets added when the object becomes dynamic.
        """
        return np.asarray(self.sim.get_base_position("object"), dtype=float)[:2]

    def _get_obs(self):
        """The 21-dim observation, in the layout documented in ``initial_state.py``.

        Returns:
            np.ndarray: Shape ``(OBS_DIM,)``, float32.
        """
        obj = self._object_xy()
        tip = self._tip_xy()
        target = np.asarray(self._task.target, dtype=float)[:2]
        return np.concatenate([
            self.robot.get_obs(),                        # 0:9
            self.scene.normalise_point(obj),             # 9:11
            self.scene.normalise_delta(obj - tip),       # 11:13
            self.scene.normalise_delta(target - obj),    # 13:15
            task_mod.encode(self._task),                 # 15:21
        ]).astype(np.float32)

    def _info(self, tip, obj):
        return {
            "is_success": task_mod.success(self._task, tip, obj),
            "target": np.asarray(self._task.target, dtype=float).copy(),
        }

    # -- gymnasium API ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Draw ``g ~ p(g)`` and ``xi ~ p(xi | g)``, then place the arm at ``xi``.

        Args:
            seed (int, optional): Reseeds ``self.np_random``.
            options (dict, optional): ``"task"`` and/or ``"xi"`` pin the draw. Used
                by the sim-vs-analytic test, which needs both fixed to compare
                against ``initial_state.h``, and by evaluation, which has to
                stratify tasks by ``task_space.coverage`` band rather than take
                what ``p(g)`` hands it.

        Returns:
            tuple: ``(observation, info)``.
        """
        super().reset(seed=seed)
        options = options or {}

        task = options.get("task")
        if task is None:
            task = task_mod.sample_task(self.np_random, self.task_id)
        xi = options.get("xi")
        if xi is None:
            xi = initial_state.sample_xi(self.np_random, task)

        self._task, self._xi, self._elapsed = task, xi, 0

        # Deliberately not robot.reset(): sample_xi draws the hand pose from the same
        # box over the same yaw range, so routing through it makes this reset and the
        # xi that h is evaluated at the *same draw* rather than two implementations
        # that agree. set_se2 re-seeds from the neutral pose and runs IK_ROUNDS, so
        # the pose is path-independent, and teleports via resetJointState, which
        # zeroes joint velocities -- which is what makes h's all-zero HAND_VEL block
        # exact rather than approximate. Nothing here may call sim.step(), or that
        # stops being true.
        with self.sim.no_rendering():
            self.robot.set_se2(xi.hand_se2[:2], float(xi.hand_se2[2]))
            self._place_markers(task, xi)

        return self._get_obs(), self._info(self._tip_xy(), self._object_xy())

    def step(self, action):
        """Advance one control step.

        ``terminated`` is unconditionally False. Success is reported in ``info`` and
        the episode always runs the full ``TaskConfig.HORIZON``: stopping early
        truncates the accumulating negative reward, which makes ``V`` jump
        discontinuously at the ``rho`` boundary and puts success and failure returns
        on different scales. ``V`` is read as an energy over designs, so that
        comparability matters more than the wall-clock an early exit would save.

        Args:
            action (np.ndarray): ``(dx, dy, dyaw)``, each in [-1, 1].

        Returns:
            tuple: ``(observation, reward, terminated, truncated, info)``.
        """
        self.robot.set_action(action)
        self.sim.step()
        self._elapsed += 1

        tip, obj = self._tip_xy(), self._object_xy()
        # Metres, not normalised units: task.reward is defined on the physical
        # distances the success tolerance rho is also expressed in.
        reward = task_mod.reward(self._task, tip, obj)
        truncated = self._elapsed >= TaskConfig.HORIZON
        return self._get_obs(), reward, False, truncated, self._info(tip, obj)

    def render(self):
        """Returns an RGB array in ``rgb_array`` mode, else None."""
        return self.sim.render()

    def close(self):
        self.sim.close()

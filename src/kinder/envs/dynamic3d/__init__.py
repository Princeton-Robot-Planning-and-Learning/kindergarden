"""Dynamic (MuJoCo) 3D environments.

Only lightweight, dependency-free declarations belong here: the gym registration
in :mod:`kinder` reads them without importing mujoco, which is optional.
"""

#: Maps the robot named first in a task JSON's "robots" dict to the env class
#: implementing it. Read both by the gym registration and by
#: :mod:`kinder.envs.dynamic3d.task_families`, whose family classes must subclass
#: the same env class their tasks declare.
ROBOT_ENV_CLASSES: dict[str, str] = {
    "tidybot": "TidyBot3D",
    "fr3": "Franka3D",
    "rby1a": "RBY1A3D",
}

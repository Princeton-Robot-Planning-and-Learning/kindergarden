"""Kinematic 3D environments built on prpl_kinematics instead of pybullet_helpers.

These environments mirror the kinematic3d ones in behavior, but model the robot as a
``prpl_kinematics.robots.Robot`` over a general ``KinematicTree``, with PyBullet demoted
to a rendering and collision backend. prpl_kinematics is an optional dependency
(``pip install kindergarden[prpl-kinematics]``), so no module outside this subpackage may
import from here at module scope.
"""

# Third-party assets

`realistic_arm_limits_model.npz` holds the weights of the arm reachability
classifier used by `RealisticArmJointLimitsModel` in `kinder/envs/dynamic3d/limbs.py`.
They come from Assistive Gym, where they ship as
`assistive_gym/envs/assets/realistic_arm_limits_model.h5`, converted from Keras HDF5
to `.npz` so that reading them needs neither Keras nor TensorFlow. The values are
unchanged.

The model is the pose-dependent joint limit model of Akhter and Black (CVPR 2015),
refitted as a network by Jiang and Liu (ICRA 2018, arXiv:1709.08685), and described in
Erickson et al., Assistive Gym (ICRA 2020, arXiv:1910.04700).

Assistive Gym is distributed under the MIT License:

    Copyright (c) 2019 Healthcare Robotics Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

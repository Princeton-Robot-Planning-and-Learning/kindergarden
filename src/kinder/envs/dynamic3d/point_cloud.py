"""Point cloud generation for Dynamic3D MuJoCo environments.

Generates dense, colored 3D point clouds from depth + RGB rendering across
one or more named cameras. The per-pixel back-projection uses each camera's
intrinsics (from cam_fovy) and extrinsics (cam_xpos / cam_xmat) live from
the simulation data so point clouds track the current physics state.

Typical usage
-------------
::

    env = kinder.make("kinder/BalanceBeam3D-o3-v0")
    env.reset(seed=0)

    sim = get_sim_from_env(env)
    pc = generate_scene_point_cloud(sim)   # uses all env cameras

    # pc.xyz  -> (N, 3) float32 world-frame positions
    # pc.rgb  -> (N, 3) uint8  colours  [0, 255]

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import mujoco  # type: ignore
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------


@dataclass
class PointCloud:
    """Merged, world-frame coloured point cloud from one or more cameras.

    Attributes:
        xyz: (N, 3) float32 array of world-frame XYZ positions.
        rgb: (N, 3) uint8 array of RGB colours in [0, 255].
        camera_indices: (N,) int array mapping each point to its source
            camera index in ``camera_names``.
        camera_names: Ordered list of camera names that contributed points.
    """

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    camera_indices: NDArray[np.int32]
    camera_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def filter_by_camera(self, name: str) -> "PointCloud":
        """Return a new PointCloud with only the points from *name*."""
        if name not in self.camera_names:
            raise ValueError(
                f"Camera '{name}' not in point cloud. "
                f"Available cameras: {self.camera_names}"
            )
        idx = self.camera_names.index(name)
        mask = self.camera_indices == idx
        return PointCloud(
            xyz=self.xyz[mask],
            rgb=self.rgb[mask],
            camera_indices=self.camera_indices[mask],
            camera_names=[name],
        )

    def to_dict(self) -> dict[str, NDArray]:
        """Serialise to a plain dict (e.g. for np.savez)."""
        return {
            "xyz": self.xyz,
            "rgb": self.rgb,
            "camera_indices": self.camera_indices,
        }


# ---------------------------------------------------------------------------
# Intrinsics / extrinsics helpers
# ---------------------------------------------------------------------------


def get_camera_intrinsics(
    model: mujoco.MjModel,  # type: ignore[name-defined]
    cam_id: int,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    """Return the 3×3 intrinsic matrix K for a MuJoCo fixed camera.

    MuJoCo uses a single vertical field-of-view (``cam_fovy``, in degrees)
    and assumes square pixels, so ``fx == fy``.

    Args:
        model: The MuJoCo model.
        cam_id: Zero-based camera index.
        width: Render width in pixels.
        height: Render height in pixels.

    Returns:
        K: 3×3 float64 intrinsics matrix::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """
    fovy_deg: float = float(model.cam_fovy[cam_id])
    fovy_rad = math.radians(fovy_deg)
    fy = height / (2.0 * math.tan(fovy_rad / 2.0))
    fx = fy  # MuJoCo cameras have square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def get_camera_extrinsics(
    data: mujoco.MjData,  # type: ignore[name-defined]
    cam_id: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the camera-to-world rotation and translation.

    MuJoCo's ``cam_xmat`` stores the camera frame orientation as a 9-element
    row-major rotation matrix whose *columns* are the camera X, Y, Z axes
    expressed in world coordinates (camera-to-world, same convention as
    ``xmat`` for bodies).  The camera Z axis points *backward* (away from
    the scene).

    Args:
        data: MuJoCo simulation data (after ``mj_forward``).
        cam_id: Zero-based camera index.

    Returns:
        R_c2w: (3, 3) rotation matrix, camera → world.
        t_w: (3,) camera position in world frame.
    """
    R_c2w = data.cam_xmat[cam_id].reshape(3, 3).copy().astype(np.float64)
    t_w = data.cam_xpos[cam_id].copy().astype(np.float64)
    return R_c2w, t_w


def depth_buffer_to_metric(
    depth_raw: NDArray[np.float32],
    model: mujoco.MjModel,  # type: ignore[name-defined]
) -> NDArray[np.float64]:
    """Convert a raw MuJoCo depth buffer to metric distances (metres).

    MuJoCo uses the standard OpenGL non-linear depth buffer where the stored
    value ``d ∈ [0, 1]`` relates to metric depth ``z`` via::

        z = znear * zfar / (zfar − d * (zfar − znear))

    where ``znear = vis.map.znear * stat.extent`` and similarly for
    ``zfar``.

    Args:
        depth_raw: (H, W) float32 array of raw depth buffer values in [0, 1].
        model: The MuJoCo model (needed for clipping-plane parameters).

    Returns:
        z_metric: (H, W) float64 array of metric distances along the optical
            axis.  Pixels at or beyond the far plane are mapped to ``inf``.
    """
    extent: float = float(model.stat.extent)
    znear: float = float(model.vis.map.znear) * extent
    zfar: float = float(model.vis.map.zfar) * extent
    d = depth_raw.astype(np.float64)
    denom = zfar - d * (zfar - znear)
    # Guard against divide-by-zero (shouldn't happen within [0,1] range)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 0.0, znear * zfar / denom, np.inf)
    return z


# ---------------------------------------------------------------------------
# Per-camera back-projection
# ---------------------------------------------------------------------------


def rgbd_to_point_cloud(
    rgb: NDArray[np.uint8],
    depth_raw: NDArray[np.float32],
    cam_id: int,
    model: mujoco.MjModel,  # type: ignore[name-defined]
    data: mujoco.MjData,  # type: ignore[name-defined]
    *,
    max_depth: float = 10.0,
    min_depth: float = 0.01,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Back-project an RGBD image pair into a world-frame point cloud.

    Pixels whose metric depth falls outside [``min_depth``, ``max_depth``]
    are discarded.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth_raw: (H, W) float32 raw MuJoCo depth buffer (values in [0, 1]).
            The image must already be flipped vertically (as returned by
            ``MjSim.render``).
        cam_id: Zero-based camera index in the model.
        model: MuJoCo model (for intrinsics and depth conversion).
        data: MuJoCo data (for extrinsics).
        max_depth: Discard points farther than this distance (metres).
        min_depth: Discard points closer than this distance (metres).

    Returns:
        xyz_world: (N, 3) float32 positions in world frame.
        rgb_points: (N, 3) uint8 colours.
    """
    height, width = depth_raw.shape
    K = get_camera_intrinsics(model, cam_id, width, height)
    R_c2w, t_w = get_camera_extrinsics(data, cam_id)

    z_metric = depth_buffer_to_metric(depth_raw, model)  # (H, W)

    # Build pixel grid
    u = np.arange(width, dtype=np.float64)  # (W,)
    v = np.arange(height, dtype=np.float64)  # (H,)
    uu, vv = np.meshgrid(u, v)  # (H, W) each

    # Valid mask
    mask = (z_metric >= min_depth) & (z_metric <= max_depth) & np.isfinite(z_metric)
    if not np.any(mask):
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.empty((0, 3), dtype=np.uint8)

    z = z_metric[mask]  # (N,)
    u_m = uu[mask]  # (N,)
    v_m = vv[mask]  # (N,)

    # Back-project to camera frame.
    # MuJoCo camera convention: X = right, Y = up, Z = backward.
    # The optical axis points in the -Z direction.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u_m - cx) / fx * z
    y_cam = -(v_m - cy) / fy * z  # flip: image v increases down, camera Y up
    z_cam = -z  # camera looks in -Z

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Transform to world frame: p_w = R_c2w @ p_c + t_w
    xyz_world = (pts_cam @ R_c2w.T + t_w).astype(np.float32)

    rgb_points = rgb[mask].astype(np.uint8)
    return xyz_world, rgb_points


# ---------------------------------------------------------------------------
# MjSim-level API
# ---------------------------------------------------------------------------


def generate_scene_point_cloud(
    sim: "MjSim",  # type: ignore[name-defined]  # kinder MjSim
    camera_names: Sequence[str] | None = None,
    *,
    width: int = 640,
    height: int = 480,
    max_depth: float = 10.0,
    min_depth: float = 0.01,
) -> PointCloud:
    """Generate a merged world-frame point cloud from one or more cameras.

    Renders each requested camera's RGB + depth image, back-projects the
    valid pixels into world-frame 3-D positions, and concatenates the
    results into a single :class:`PointCloud`.

    Args:
        sim: A ``kinder`` :class:`MjSim` instance (already reset and
            forwarded).
        camera_names: Camera names to use.  Pass ``None`` to use every
            named camera in the model.
        width: Render width in pixels.
        height: Render height in pixels.
        max_depth: Maximum metric depth to keep (metres).
        min_depth: Minimum metric depth to keep (metres).

    Returns:
        A :class:`PointCloud` containing the merged scene geometry.

    Raises:
        ValueError: If a requested camera name is not found in the model.
    """
    mj_model: mujoco.MjModel = sim.model.mj_model  # type: ignore[name-defined]
    mj_data: mujoco.MjData = sim.data.mj_data  # type: ignore[name-defined]

    # Resolve camera names → camera IDs
    name2id: dict[str, int] = {}
    for i in range(mj_model.ncam):
        name = mujoco.mj_id2name(  # pylint: disable=no-member
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i  # pylint: disable=no-member
        )
        if name is not None:
            name2id[name] = i

    if camera_names is None:
        # Use all named cameras
        cam_items = list(name2id.items())
    else:
        cam_items = []
        for cname in camera_names:
            if cname not in name2id:
                raise ValueError(
                    f"Camera '{cname}' not found in model. "
                    f"Available cameras: {list(name2id.keys())}"
                )
            cam_items.append((cname, name2id[cname]))

    if not cam_items:
        empty_f = np.empty((0, 3), dtype=np.float32)
        empty_u = np.empty((0, 3), dtype=np.uint8)
        empty_i = np.empty((0,), dtype=np.int32)
        return PointCloud(xyz=empty_f, rgb=empty_u, camera_indices=empty_i)

    all_xyz: list[NDArray[np.float32]] = []
    all_rgb: list[NDArray[np.uint8]] = []
    all_idx: list[NDArray[np.int32]] = []
    resolved_names: list[str] = []

    ctx = sim._render_context_offscreen  # pylint: disable=protected-access

    for local_idx, (cname, cam_id) in enumerate(cam_items):
        ctx.render(width=width, height=height, camera_id=cam_id)
        rgb_depth = ctx.read_pixels(width, height, depth=True)
        assert isinstance(rgb_depth, tuple)
        rgb_img, depth_img = rgb_depth
        assert depth_img is not None

        xyz, rgb = rgbd_to_point_cloud(
            rgb_img,
            depth_img,
            cam_id,
            mj_model,
            mj_data,
            max_depth=max_depth,
            min_depth=min_depth,
        )

        if len(xyz) > 0:
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_idx.append(np.full(len(xyz), local_idx, dtype=np.int32))
        resolved_names.append(cname)

    if not all_xyz:
        empty_f = np.empty((0, 3), dtype=np.float32)
        empty_u = np.empty((0, 3), dtype=np.uint8)
        empty_i = np.empty((0,), dtype=np.int32)
        return PointCloud(
            xyz=empty_f,
            rgb=empty_u,
            camera_indices=empty_i,
            camera_names=resolved_names,
        )

    return PointCloud(
        xyz=np.concatenate(all_xyz, axis=0),
        rgb=np.concatenate(all_rgb, axis=0),
        camera_indices=np.concatenate(all_idx, axis=0),
        camera_names=resolved_names,
    )


# ---------------------------------------------------------------------------
# Complete (occlusion-free) point cloud via mesh sampling
# ---------------------------------------------------------------------------


def _quat_to_matrix(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a MuJoCo (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_from_pos_quat(
    pos: NDArray[np.float64],
    quat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a 4x4 homogeneous transform from a position and (w, x, y, z) quat."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_matrix(quat)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _generate_box_mesh(model: "mujoco.MjModel", geom_id: int):  # type: ignore[name-defined]
    """Triangle mesh for an ``mjGEOM_BOX`` in its local frame."""
    import trimesh  # pylint: disable=import-outside-toplevel  # type: ignore[import]

    half_extents = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = signs * half_extents
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],
            [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4],
            [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _generate_cylinder_mesh(
    model: "mujoco.MjModel",  # type: ignore[name-defined]
    geom_id: int,
    n_segments: int = 20,
):
    """Triangle mesh for an ``mjGEOM_CYLINDER`` in its local frame."""
    import trimesh  # pylint: disable=import-outside-toplevel  # type: ignore[import]

    radius = float(model.geom_size[geom_id][0])
    half_height = float(model.geom_size[geom_id][1])

    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    bottom_ring = np.stack([x, y, np.full_like(x, -half_height)], axis=1)
    top_ring = np.stack([x, y, np.full_like(x, half_height)], axis=1)
    center_bottom = np.array([[0.0, 0.0, -half_height]])
    center_top = np.array([[0.0, 0.0, half_height]])
    vertices = np.vstack([bottom_ring, top_ring, center_bottom, center_top])

    i_center_bottom = 2 * n_segments
    i_center_top = 2 * n_segments + 1

    faces: list[list[int]] = []
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.append([i, i + n_segments, i_next + n_segments])
        faces.append([i, i_next + n_segments, i_next])
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.append([i_center_bottom, i_next, i])
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.append([i_center_top, i + n_segments, i_next + n_segments])

    return trimesh.Trimesh(
        vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False
    )


def _generate_sphere_mesh(
    model: "mujoco.MjModel",  # type: ignore[name-defined]
    geom_id: int,
    subdivisions: int = 3,
):
    """Triangle mesh for an ``mjGEOM_SPHERE`` in its local frame."""
    import trimesh  # pylint: disable=import-outside-toplevel  # type: ignore[import]

    radius = float(model.geom_size[geom_id][0])
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def _generate_capsule_mesh(
    model: "mujoco.MjModel",  # type: ignore[name-defined]
    geom_id: int,
    segments: int = 20,
    subdivisions: int = 2,
):
    """Triangle mesh for an ``mjGEOM_CAPSULE`` in its local frame.

    A capsule consists of a cylinder of length ``2 * half_height`` along the
    Z axis with hemispherical caps at each end.
    """
    import trimesh  # pylint: disable=import-outside-toplevel  # type: ignore[import]

    radius = float(model.geom_size[geom_id][0])
    half_height = float(model.geom_size[geom_id][1])

    cylinder = trimesh.creation.cylinder(
        radius=radius, height=2.0 * half_height, sections=segments
    )
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

    # Top hemisphere (z >= 0), translated to +half_height
    top_mask = sphere.vertices[:, 2] >= 0
    top_face_mask = np.all(top_mask[sphere.faces], axis=1)
    top_face_idx = sphere.faces[top_face_mask]
    top_vert_idx = np.where(top_mask)[0]
    remap = {old: new for new, old in enumerate(top_vert_idx)}
    top_faces_new = np.vectorize(remap.get)(top_face_idx)
    top = trimesh.Trimesh(
        vertices=sphere.vertices[top_vert_idx],
        faces=top_faces_new,
        process=False,
    )
    top.apply_translation([0.0, 0.0, half_height])

    bottom = top.copy()
    bottom.apply_scale([1.0, 1.0, -1.0])

    return trimesh.util.concatenate([cylinder, top, bottom])


def _build_geom_mesh(
    model: "mujoco.MjModel",  # type: ignore[name-defined]
    geom_id: int,
):
    """Return a ``trimesh.Trimesh`` for a geom, or ``None`` if unsupported.

    Supports ``mjGEOM_MESH``, ``mjGEOM_BOX``, ``mjGEOM_CYLINDER``,
    ``mjGEOM_SPHERE`` and ``mjGEOM_CAPSULE``.  Planes, height fields and
    other primitives are silently skipped.
    """
    import trimesh  # pylint: disable=import-outside-toplevel  # type: ignore[import]

    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):  # pylint: disable=no-member
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        v_start = int(model.mesh_vertadr[mesh_id])
        v_n = int(model.mesh_vertnum[mesh_id])
        f_start = int(model.mesh_faceadr[mesh_id])
        f_n = int(model.mesh_facenum[mesh_id])
        verts = np.asarray(model.mesh_vert[v_start : v_start + v_n], dtype=np.float64)
        faces = np.asarray(model.mesh_face[f_start : f_start + f_n], dtype=np.int64)
        if len(verts) == 0 or len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):  # pylint: disable=no-member
        return _generate_box_mesh(model, geom_id)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):  # pylint: disable=no-member
        return _generate_cylinder_mesh(model, geom_id)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):  # pylint: disable=no-member
        return _generate_sphere_mesh(model, geom_id)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):  # pylint: disable=no-member
        return _generate_capsule_mesh(model, geom_id)
    return None


def _geom_color_uint8(
    model: "mujoco.MjModel",  # type: ignore[name-defined]
    geom_id: int,
) -> NDArray[np.uint8]:
    """Look up an RGB colour (0-255) for a geom, preferring its material."""
    matid = int(model.geom_matid[geom_id])
    if matid >= 0:
        rgba = np.asarray(model.mat_rgba[matid], dtype=np.float64)
    else:
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64)
    return np.clip(rgba[:3] * 255.0, 0, 255).astype(np.uint8)


def generate_complete_scene_point_cloud(
    sim: "MjSim",  # type: ignore[name-defined]
    *,
    num_points_per_geom: int = 500,
    seed: int | None = None,
) -> PointCloud:
    """Generate an occlusion-free, world-frame point cloud by sampling all geoms.

    Unlike :func:`generate_scene_point_cloud`, this function does **not** use
    cameras, depth buffers, or RGB rendering.  It iterates over every geom
    in the model, builds a triangle mesh for it
    (``MESH``/``BOX``/``CYLINDER``/``SPHERE``/``CAPSULE``), samples
    ``num_points_per_geom`` surface points, and transforms them into the
    world frame using the geom's current body pose
    (``data.xpos`` / ``data.xmat``) and the geom's local offset
    (``model.geom_pos`` / ``model.geom_quat``).

    Each point is coloured using the geom's material colour (or
    ``geom_rgba`` if no material is set).  ``camera_names`` and
    ``camera_indices`` are repurposed here to identify the **source geom**
    for each point.

    Args:
        sim: A ``kinder`` :class:`MjSim` instance (already reset and
            forwarded).
        num_points_per_geom: Number of surface points sampled per geom.
        seed: Optional RNG seed for reproducible sampling.

    Returns:
        A :class:`PointCloud` whose ``camera_names`` is a list of geom
        labels (``"<name>_geom<id>"``) and ``camera_indices`` maps each
        point to its index in that list.
    """
    try:
        import trimesh  # pylint: disable=import-outside-toplevel,unused-import  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "trimesh is required for complete point cloud generation. "
            "Install it with: pip install trimesh"
        ) from exc

    mj_model: mujoco.MjModel = sim.model.mj_model  # type: ignore[name-defined]
    mj_data: mujoco.MjData = sim.data.mj_data  # type: ignore[name-defined]

    rng = np.random.default_rng(seed)

    all_xyz: list[NDArray[np.float32]] = []
    all_rgb: list[NDArray[np.uint8]] = []
    all_idx: list[NDArray[np.int32]] = []
    geom_labels: list[str] = []

    for geom_id in range(mj_model.ngeom):
        mesh = _build_geom_mesh(mj_model, geom_id)
        if mesh is None:
            continue

        try:
            samples, _ = mesh.sample(num_points_per_geom, return_index=True)
        except Exception:  # pylint: disable=broad-except
            verts = np.asarray(mesh.vertices)
            if len(verts) == 0:
                continue
            pick = rng.integers(0, len(verts), size=num_points_per_geom)
            samples = verts[pick]
        points_local = np.asarray(samples, dtype=np.float64)

        body_id = int(mj_model.geom_bodyid[geom_id])
        body_pos = np.asarray(mj_data.xpos[body_id], dtype=np.float64)
        body_mat = np.asarray(mj_data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        body_T = np.eye(4, dtype=np.float64)
        body_T[:3, :3] = body_mat
        body_T[:3, 3] = body_pos

        local_T = _transform_from_pos_quat(
            np.asarray(mj_model.geom_pos[geom_id], dtype=np.float64),
            np.asarray(mj_model.geom_quat[geom_id], dtype=np.float64),
        )
        T = body_T @ local_T

        homog = np.hstack([points_local, np.ones((points_local.shape[0], 1))])
        world = (homog @ T.T)[:, :3].astype(np.float32)

        rgb255 = _geom_color_uint8(mj_model, geom_id)
        rgb_arr = np.tile(rgb255, (len(world), 1)).astype(np.uint8)

        idx = len(geom_labels)
        all_xyz.append(world)
        all_rgb.append(rgb_arr)
        all_idx.append(np.full(len(world), idx, dtype=np.int32))

        geom_name = mujoco.mj_id2name(  # pylint: disable=no-member
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id  # pylint: disable=no-member
        )
        label = f"{geom_name}_geom{geom_id}" if geom_name else f"geom{geom_id}"
        geom_labels.append(label)

    if not all_xyz:
        return PointCloud(
            xyz=np.empty((0, 3), dtype=np.float32),
            rgb=np.empty((0, 3), dtype=np.uint8),
            camera_indices=np.empty((0,), dtype=np.int32),
            camera_names=[],
        )

    return PointCloud(
        xyz=np.concatenate(all_xyz, axis=0),
        rgb=np.concatenate(all_rgb, axis=0),
        camera_indices=np.concatenate(all_idx, axis=0),
        camera_names=geom_labels,
    )


# ---------------------------------------------------------------------------
# Convenience: extract sim from a wrapped kinder env
# ---------------------------------------------------------------------------


def get_sim_from_env(env: Any) -> "MjSim":  # type: ignore[name-defined]
    """Extract the underlying :class:`MjSim` from a wrapped kinder env.

    Works with the standard ``kinder.make(...)`` wrapper stack::

        kinder_wrapper
          └─ ObjectCentricRobotEnv (_object_centric_env)
               └─ RobotEnv         (_robot_env)
                    └─ MujocoEnv   (sim)

    Args:
        env: Any gymnasium-wrapped kinder environment.

    Returns:
        The ``MjSim`` instance (guaranteed non-None after a reset).

    Raises:
        RuntimeError: If the expected attributes are not found, or if the
            sim has not been initialised (i.e. no reset has been called yet).
    """
    unwrapped = env.unwrapped

    # Path: object_centric_env → _robot_env → sim
    if hasattr(unwrapped, "_object_centric_env"):
        # pylint: disable=protected-access
        robot_env = unwrapped._object_centric_env._robot_env
    elif hasattr(unwrapped, "_robot_env"):
        robot_env = unwrapped._robot_env  # pylint: disable=protected-access
    else:
        raise RuntimeError(
            "Cannot locate a robot env inside the provided environment. "
            "Expected either '_object_centric_env._robot_env' or '_robot_env' "
            f"on {type(unwrapped).__name__}."
        )

    sim = robot_env.sim
    if sim is None:
        raise RuntimeError(
            "sim is None — call env.reset() before generating a point cloud."
        )
    return sim

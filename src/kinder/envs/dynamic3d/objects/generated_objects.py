"""Generated mesh objects for dynamic3d environments."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv
from kinder.envs.dynamic3d.object_types import MujocoMovableObjectType
from kinder.envs.dynamic3d.objects.base import MujocoObject, register_object
from kinder.envs.dynamic3d.objects.utils import save_mesh


@register_object(name="generated_bowl")
class GeneratedBowl(MujocoObject):
    """A procedurally generated bowl object for TidyBot environments."""

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a GeneratedBowl object.

        Args:
            name: Name of the bowl body in the XML
            env: Reference to the environment
            options: Dictionary of bowl options:
                - outer_radius: Outer radius at rim (default: 0.05m)
                - inner_radius: Inner radius at rim (default: 0.045m)
                - height: Height/depth of the bowl (default: 0.025m)
                - wall_thickness: Bowl wall thickness at bottom (default: 0.003m)
                - radial_segments: Segments around circumference (default: 32)
                - vertical_segments: Segments from rim to bottom (default: 16)
                - rgba: Color as string or [r, g, b, a] (default: ".5 .5 .5 1")
                - mass: Mass of the bowl (default: 0.05)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Bowl parameters
        self.outer_radius: float = float(self.options.get("outer_radius", 0.05))
        self.inner_radius: float = float(self.options.get("inner_radius", 0.045))
        self.height: float = float(self.options.get("height", 0.025))
        self.wall_thickness: float = float(self.options.get("wall_thickness", 0.003))
        self.radial_segments: int = int(self.options.get("radial_segments", 32))
        self.vertical_segments: int = int(self.options.get("vertical_segments", 16))

        # Handle rgba parameter
        rgba = self.options.get("rgba", ".5 .5 .5 1")
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        self.mass: float = float(self.options.get("mass", 0.05))

        # Generate mesh and create temporary OBJ file
        self.mesh_file = self._generate_and_save_mesh()
        self.mesh_name = f"{self.name}_bowl_mesh"

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            self._create_regions()

    def _generate_and_save_mesh(self) -> str:
        """Generate bowl mesh and save to a temporary OBJ file.

        Returns:
            Path to the generated OBJ file
        """
        # Generate mesh vertices and faces
        vertices, faces = self._generate_bowl_mesh()

        # Define directory for temporary meshes
        mesh_dir = Path(__file__).parents[1] / "models" / "assets" / ".tmp"

        # Save mesh using utility function
        return save_mesh(vertices, faces, mesh_dir)

    def _generate_bowl_mesh(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a bowl mesh with specified dimensions.

        Returns:
            Tuple of (vertices, faces) arrays
        """
        vertices_list = []
        faces_list = []

        # Generate outer surface (true hemisphere shape)
        for i in range(self.vertical_segments + 1):
            # Angle from vertical: pi/2 at rim (i=0), 0 at bottom
            theta = (np.pi / 2) * (1 - i / self.vertical_segments)

            # For a true hemisphere bowl:
            # At i=0 (rim/top): theta=pi/2, r=outer_radius, z=0
            # At i=vertical_segments (bottom): theta=0, r=0, z=-height
            r = self.outer_radius * np.sin(theta)
            z = -self.height * np.cos(theta)

            for j in range(self.radial_segments):
                phi = (j / self.radial_segments) * 2 * np.pi
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                vertices_list.append([x, y, z])

        # Generate inner surface (slightly smaller hemisphere)
        for i in range(self.vertical_segments + 1):
            theta = (np.pi / 2) * (1 - i / self.vertical_segments)
            r = self.inner_radius * np.sin(theta)
            z = -self.height * np.cos(theta)

            for j in range(self.radial_segments):
                phi = (j / self.radial_segments) * 2 * np.pi
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                vertices_list.append([x, y, z])

        vertices = np.array(vertices_list)

        # Create faces for outer surface
        num_outer_vertices = (self.vertical_segments + 1) * self.radial_segments
        for i in range(self.vertical_segments):
            for j in range(self.radial_segments):
                # Current quad vertices
                v0 = i * self.radial_segments + j
                v1 = i * self.radial_segments + (j + 1) % self.radial_segments
                v2 = (i + 1) * self.radial_segments + (j + 1) % self.radial_segments
                v3 = (i + 1) * self.radial_segments + j

                # Two triangles per quad (outer surface faces outward)
                faces_list.append([v0, v2, v1])
                faces_list.append([v0, v3, v2])

        # Create faces for inner surface
        for i in range(self.vertical_segments):
            for j in range(self.radial_segments):
                # Current quad vertices (offset by outer surface vertices)
                v0 = num_outer_vertices + i * self.radial_segments + j
                v1 = (
                    num_outer_vertices
                    + i * self.radial_segments
                    + (j + 1) % self.radial_segments
                )
                v2 = (
                    num_outer_vertices
                    + (i + 1) * self.radial_segments
                    + (j + 1) % self.radial_segments
                )
                v3 = num_outer_vertices + (i + 1) * self.radial_segments + j

                # Two triangles per quad (inner surface faces inward, so reverse winding)
                faces_list.append([v0, v1, v2])
                faces_list.append([v0, v2, v3])

        # Create rim (flat surface connecting outer rim to inner rim at the top)
        for j in range(self.radial_segments):
            # Outer rim vertices (i=0, at the opening)
            v0_outer = j
            v1_outer = (j + 1) % self.radial_segments

            # Inner rim vertices (i=0, at the opening)
            v0_inner = num_outer_vertices + j
            v1_inner = num_outer_vertices + (j + 1) % self.radial_segments

            # Two triangles to create flat rim surface (facing upward)
            faces_list.append([v0_outer, v1_outer, v1_inner])
            faces_list.append([v0_outer, v1_inner, v0_inner])

        faces = np.array(faces_list)

        return vertices, faces

    def get_assets(self) -> list[ET.Element]:
        """Get the asset elements (mesh) for this bowl.

        Returns:
            List of ET.Element containing mesh asset
        """
        # Create mesh asset element
        mesh_elem = ET.Element("mesh")
        mesh_elem.set("file", self.mesh_file)
        mesh_elem.set("name", self.mesh_name)

        return [mesh_elem]

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this bowl.

        Returns:
            ET.Element representing the bowl body
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add geom element with mesh reference (mesh will be added to assets)
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=self.mesh_name,
            rgba=self.rgba,
            mass=str(self.mass),
        )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this bowl.

        Returns:
            Tuple of (width, depth, height) for the bounding box
        """
        # Bowl dimensions: diameter x diameter x height
        diameter = 2 * self.outer_radius
        return (diameter, diameter, self.height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a bowl given its position and config.

        Args:
            pos: Position of the bowl as [x, y, z] array
            object_config: Dictionary containing bowl configuration

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract bowl parameters
        outer_radius = float(object_config.get("outer_radius", 0.05))
        height = float(object_config.get("height", 0.025))

        # Half-extents
        half_diameter = outer_radius
        half_height = height / 2

        return [
            float(pos[0]) - half_diameter,  # x_min
            float(pos[1]) - half_diameter,  # y_min
            float(pos[2]) - half_height,  # z_min
            float(pos[0]) + half_diameter,  # x_max
            float(pos[1]) + half_diameter,  # y_max
            float(pos[2]) + half_height,  # z_max
        ]

    def __str__(self) -> str:
        """String representation of the bowl."""
        return (
            f"GeneratedBowl(name='{self.name}', "
            f"outer_radius={self.outer_radius}, inner_radius={self.inner_radius}, "
            f"height={self.height})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the bowl."""
        return (
            f"GeneratedBowl(name='{self.name}', joint_name='{self.joint_name}', "
            f"outer_radius={self.outer_radius}, inner_radius={self.inner_radius}, "
            f"height={self.height}, mass={self.mass})"
        )


@register_object(name="generated_seesaw")
class GeneratedSeesaw(MujocoObject):
    """A procedurally generated seesaw object for dynamic3d environments.

    The seesaw consists of:
    - A long, narrow beam (board) that can tilt
    - A triangular pivot (fulcrum) that supports the beam at its center

    The beam can rotate freely around the pivot point via a hinge joint.
    """

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a GeneratedSeesaw object.

        Args:
            name: Name of the seesaw body in the XML
            env: Reference to the environment
            options: Dictionary of seesaw options:
                - beam_length: Length of the beam (default: 0.4m)
                - beam_width: Width of the beam (default: 0.06m)
                - beam_thickness: Thickness of the beam (default: 0.01m)
                - beam_clearance: Gap between pivot apex and beam center (
                    default: 0.002m)
                - pivot_height: Height of the pivot/fulcrum (default: 0.04m)
                - pivot_width: Width of the pivot base (default: 0.04m)
                - beam_rgba: Color of the beam (default: "0.6 0.4 0.2 1")
                - pivot_rgba: Color of the pivot (default: "0.4 0.4 0.4 1")
                - beam_friction: Contact friction tuple/string for the beam
                  (default: [1.4, 0.02, 0.002])
                - pivot_friction: Contact friction tuple/string for the pivot
                  (default: [1.2, 0.02, 0.002])
                - beam_mass: Mass of the beam (default: 0.1)
                - pivot_mass: Mass of the pivot (default: 0.2)
                - damping: Joint damping coefficient (default: 0.015)
                - hinge_frictionloss: Coulomb friction at hinge (default: 0.02)
                - hinge_stiffness: Torsional spring around 0 deg (default: 1.0)
                - hinge_range: Optional angular limits [min, max] in radians
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Beam parameters
        self.beam_length: float = float(self.options.get("beam_length", 0.4))
        self.beam_width: float = float(self.options.get("beam_width", 0.06))
        self.beam_thickness: float = float(self.options.get("beam_thickness", 0.01))
        self.beam_clearance: float = float(self.options.get("beam_clearance", 0.002))

        # Pivot (fulcrum) parameters
        self.pivot_height: float = float(self.options.get("pivot_height", 0.04))
        self.pivot_width: float = float(self.options.get("pivot_width", 0.04))

        # Handle beam rgba parameter
        beam_rgba = self.options.get("beam_rgba", "0.6 0.4 0.2 1")
        if isinstance(beam_rgba, str):
            self.beam_rgba = beam_rgba
        else:
            self.beam_rgba = " ".join(str(x) for x in beam_rgba)

        # Handle pivot rgba parameter
        pivot_rgba = self.options.get("pivot_rgba", "0.4 0.4 0.4 1")
        if isinstance(pivot_rgba, str):
            self.pivot_rgba = pivot_rgba
        else:
            self.pivot_rgba = " ".join(str(x) for x in pivot_rgba)

        # Contact/friction parameters for beam and pivot
        default_beam_friction = [1.4, 0.02, 0.002]
        beam_friction = self.options.get("beam_friction", default_beam_friction)
        if isinstance(beam_friction, str):
            self.beam_friction = beam_friction
        else:
            self.beam_friction = " ".join(str(x) for x in beam_friction)

        default_pivot_friction = [1.2, 0.02, 0.002]
        pivot_friction = self.options.get("pivot_friction", default_pivot_friction)
        if isinstance(pivot_friction, str):
            self.pivot_friction = pivot_friction
        else:
            self.pivot_friction = " ".join(str(x) for x in pivot_friction)

        self.beam_mass: float = float(self.options.get("beam_mass", 0.1))
        self.pivot_mass: float = float(self.options.get("pivot_mass", 0.2))
        self.damping: float = float(self.options.get("damping", 0.015))

        # Hinge stability parameters (make balancing less twitchy)
        self.hinge_frictionloss: float = float(
            self.options.get("hinge_frictionloss", 0.02)
        )
        self.hinge_stiffness: float = float(self.options.get("hinge_stiffness", 1.0))
        hinge_range = self.options.get("hinge_range", [-0.8, 0.8])
        if hinge_range is None:
            self.hinge_range: tuple[float, float] | None = None
        else:
            if len(hinge_range) != 2:
                raise ValueError("hinge_range must be a 2-element sequence [min, max]")
            self.hinge_range = (float(hinge_range[0]), float(hinge_range[1]))

        # Generate mesh for pivot and save to temporary OBJ file
        self.pivot_mesh_file = self._generate_and_save_pivot_mesh()
        self.pivot_mesh_name = f"{self.name}_pivot_mesh"

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            # A region attached to a seesaw may leave its ranges out, in which case it
            # covers the beam. A task that wants "these objects end up on the beam" can
            # then say so with the ordinary `on` predicate without restating geometry
            # the seesaw already owns, and without going stale when that geometry
            # changes. Rebuilt rather than mutated: the dicts come straight from the
            # task config and are shared with it.
            self.regions = {
                name: (
                    config
                    if config.get("ranges")
                    else {**config, "ranges": [self.beam_surface_range()]}
                )
                for name, config in self.regions.items()
            }
            self._create_regions()

    def beam_surface_range(self) -> list[float]:
        """The volume an object occupies when it is resting on the beam.

        Returns ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in the seesaw's own
        frame, so the beam runs along x whatever yaw the task spawns the seesaw with.

        The floor sits at the pivot rather than at the beam's top surface. The beam
        tilts, so an object near the end rides below the level surface by up to
        ``beam_length / 2 * sin(tilt)`` -- 1.6cm at the 5 degrees BalanceBeam3D calls
        balanced, more than the 1cm of slack region bounds are given. The pivot is
        still well clear of anything lying on the ground beside or beneath the beam.

        The ceiling is deliberately generous: gravity decides what is resting on the
        beam, so it only has to exclude something held far above it.
        """
        half_length = self.beam_length / 2
        half_width = self.beam_width / 2
        return [
            -half_length,
            -half_width,
            self.pivot_height,
            half_length,
            half_width,
            self.pivot_height + self.beam_clearance + half_length,
        ]

    def _generate_and_save_pivot_mesh(self) -> str:
        """Generate triangular prism pivot mesh and save to a temporary OBJ file.

        Returns:
            Path to the generated OBJ file
        """
        # Generate mesh vertices and faces
        vertices, faces = self._generate_pivot_mesh()

        # Define directory for temporary meshes
        mesh_dir = Path(__file__).parents[1] / "models" / "assets" / ".tmp"

        # Save mesh using utility function
        return save_mesh(vertices, faces, mesh_dir)

    def _generate_pivot_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a triangular prism mesh for the pivot/fulcrum.

        The pivot is a triangular prism oriented so the ridge runs along Y-axis
        (parallel to the beam width), allowing the beam to tilt along the X-axis.

        Returns:
            Tuple of (vertices, faces) arrays
        """
        # Half dimensions
        half_width = self.pivot_width / 2  # Base half-width in X
        half_depth = self.beam_width / 2  # Depth in Y (same as beam width)
        height = self.pivot_height

        # Vertices of triangular prism
        # Front face (y = -half_depth): triangle with apex at top
        # Back face (y = +half_depth): triangle with apex at top
        vertices = np.array(
            [
                # Front face triangle (y = -half_depth)
                [-half_width, -half_depth, 0],  # 0: bottom left
                [half_width, -half_depth, 0],  # 1: bottom right
                [0, -half_depth, height],  # 2: top apex
                # Back face triangle (y = +half_depth)
                [-half_width, half_depth, 0],  # 3: bottom left
                [half_width, half_depth, 0],  # 4: bottom right
                [0, half_depth, height],  # 5: top apex
            ]
        )

        # Faces (triangles with correct winding for outward normals)
        faces = np.array(
            [
                # Front face (facing -Y)
                [0, 2, 1],
                # Back face (facing +Y)
                [3, 4, 5],
                # Bottom face (facing -Z)
                [0, 1, 4],
                [0, 4, 3],
                # Left slope face (facing -X, +Z)
                [0, 3, 5],
                [0, 5, 2],
                # Right slope face (facing +X, +Z)
                [1, 2, 5],
                [1, 5, 4],
            ]
        )

        return vertices, faces

    def get_assets(self) -> list[ET.Element]:
        """Get the asset elements (mesh) for this seesaw.

        Returns:
            List of ET.Element containing mesh asset for the pivot
        """
        # Create mesh asset element for pivot
        mesh_elem = ET.Element("mesh")
        mesh_elem.set("file", self.pivot_mesh_file)
        mesh_elem.set("name", self.pivot_mesh_name)

        return [mesh_elem]

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this seesaw.

        The seesaw structure:
        - Base body (pivot/fulcrum) - static relative to parent
        - Child body (beam) - connected via hinge joint

        Returns:
            ET.Element representing the seesaw body hierarchy
        """
        # Create main body element (this will be positioned by the environment)
        body = ET.Element("body", name=self.name)

        # Add freejoint for the entire seesaw to be positionable
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add pivot (fulcrum) geom using generated mesh
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_pivot",
            type="mesh",
            mesh=self.pivot_mesh_name,
            rgba=self.pivot_rgba,
            mass=str(self.pivot_mass),
            friction=self.pivot_friction,
        )

        # Create beam as a child body with hinge joint
        # Position beam at the top of the pivot
        beam_body = ET.SubElement(
            body,
            "body",
            name=f"{self.name}_beam",
            pos=f"0 0 {self.pivot_height + self.beam_clearance}",
        )

        # Add hinge joint for beam rotation around Y-axis (tilt left-right)
        joint_kwargs: dict[str, str] = {
            "name": f"{self.name}_hinge",
            "type": "hinge",
            "axis": "0 1 0",  # Rotate around Y-axis
            "damping": str(self.damping),
            "frictionloss": str(self.hinge_frictionloss),
            "stiffness": str(self.hinge_stiffness),
        }
        if self.hinge_range is not None:
            joint_kwargs["range"] = f"{self.hinge_range[0]} {self.hinge_range[1]}"
        ET.SubElement(beam_body, "joint", joint_kwargs)

        # Add beam geom (box shape)
        # Beam is centered at the hinge point
        beam_half_length = self.beam_length / 2
        beam_half_width = self.beam_width / 2
        beam_half_thickness = self.beam_thickness / 2

        ET.SubElement(
            beam_body,
            "geom",
            name=f"{self.name}_beam_geom",
            type="box",
            size=f"{beam_half_length} {beam_half_width} {beam_half_thickness}",
            rgba=self.beam_rgba,
            mass=str(self.beam_mass),
            friction=self.beam_friction,
        )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this seesaw.

        Returns:
            Tuple of (width, depth, height) for the bounding box
        """
        # Seesaw dimensions: beam_length x beam_width x (pivot_height +
        # clearance + beam_thickness)
        total_height = self.pivot_height + self.beam_clearance + self.beam_thickness
        return (self.beam_length, self.beam_width, total_height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a seesaw given its position and config.

        Args:
            pos: Position of the seesaw as [x, y, z] array
            object_config: Dictionary containing seesaw configuration

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract seesaw parameters
        beam_length = float(object_config.get("beam_length", 0.4))
        beam_width = float(object_config.get("beam_width", 0.06))
        beam_thickness = float(object_config.get("beam_thickness", 0.01))
        beam_clearance = float(object_config.get("beam_clearance", 0.002))
        pivot_height = float(object_config.get("pivot_height", 0.04))

        # Half-extents
        half_length = beam_length / 2
        half_width = beam_width / 2
        total_height = pivot_height + beam_clearance + beam_thickness

        return [
            float(pos[0]) - half_length,  # x_min
            float(pos[1]) - half_width,  # y_min
            float(pos[2]),  # z_min (base at position)
            float(pos[0]) + half_length,  # x_max
            float(pos[1]) + half_width,  # y_max
            float(pos[2]) + total_height,  # z_max
        ]

    def __str__(self) -> str:
        """String representation of the seesaw."""
        return (
            f"GeneratedSeesaw(name='{self.name}', "
            f"beam_length={self.beam_length}, beam_width={self.beam_width}, "
            f"pivot_height={self.pivot_height})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the seesaw."""
        return (
            f"GeneratedSeesaw(name='{self.name}', joint_name='{self.joint_name}', "
            f"beam_length={self.beam_length}, beam_width={self.beam_width}, "
            f"beam_thickness={self.beam_thickness}, pivot_height={self.pivot_height}, "
            f"beam_mass={self.beam_mass}, pivot_mass={self.pivot_mass})"
        )

    def get_beam_tilt_angle(self) -> float:
        """Get the current tilt angle of the beam in radians.

        The hinge joint angle represents the rotation of the beam around the Y-axis.
        A positive angle means the right side (positive X) is tilted down.
        A negative angle means the left side (negative X) is tilted down.
        An angle of 0 means the beam is level/balanced.

        Returns:
            The beam tilt angle in radians.

        Raises:
            ValueError: If environment is not set.
        """
        if self.env is None:
            raise ValueError("Environment must be set to get beam tilt angle")

        assert self.env.sim is not None, "Simulation not initialized"

        # Get the hinge joint angle from the simulation
        hinge_joint_name = f"{self.name}_hinge"
        joint_qpos_addr = self.env.sim.model.get_joint_qpos_addr(hinge_joint_name)
        angle = float(self.env.sim.data.mj_data.qpos[joint_qpos_addr])
        return angle

    def get_beam_tilt_angle_degrees(self) -> float:
        """Get the current tilt angle of the beam in degrees.

        Returns:
            The beam tilt angle in degrees.

        Raises:
            ValueError: If environment is not set.
        """
        return np.degrees(self.get_beam_tilt_angle())

    def is_balanced(self, tolerance_degrees: float = 5.0) -> bool:
        """Check if the beam is balanced (within tolerance of horizontal).

        Args:
            tolerance_degrees: Maximum allowed deviation from horizontal in degrees.

        Returns:
            True if the beam tilt is within the tolerance, False otherwise.

        Raises:
            ValueError: If environment is not set.
        """
        angle_degrees = abs(self.get_beam_tilt_angle_degrees())
        return angle_degrees <= tolerance_degrees

    def is_object_on_beam(
        self,
        object_position: NDArray[np.float32],
        tolerance: float = 0.02,
    ) -> bool:
        """Check if an object is on the seesaw beam.

        Args:
            object_position: The object's position [x, y, z] in world coordinates.
            tolerance: Extra tolerance for position checks (default 0.02m).

        Returns:
            True if the object is on the seesaw beam, False otherwise.

        Raises:
            ValueError: If environment is not set.
        """
        if self.env is None:
            raise ValueError("Environment must be set to check object position")

        # Get seesaw pose. The beam runs along the seesaw's local +x, which is not
        # world +x whenever the task spawns the seesaw with a yaw (BalanceBeam3D-o3
        # pins yaw to 90 degrees), so the extent checks below have to be made in the
        # seesaw's frame. Comparing world offsets instead swaps the length and width
        # limits, and a block placed correctly along the beam reads as off it.
        seesaw_pos, seesaw_quat = self.env.get_joint_pos_quat(self.joint_name)

        # MuJoCo quaternions are (w, x, y, z); scipy wants (x, y, z, w).
        rotation = Rotation.from_quat(
            [seesaw_quat[1], seesaw_quat[2], seesaw_quat[3], seesaw_quat[0]]
        )
        # Yaw only: the beam pitches about its hinge as it tilts, and folding that
        # rotation in would shrink the footprint an object has to sit within just
        # because the beam is not level. The pivot stays upright regardless.
        yaw = rotation.as_euler("xyz")[2]
        world_offset = np.asarray(object_position, dtype=np.float64) - np.asarray(
            seesaw_pos, dtype=np.float64
        )
        local_offset = Rotation.from_euler("z", -yaw).apply(world_offset)

        beam_half_length = self.beam_length / 2
        beam_half_width = self.beam_width / 2
        min_height = seesaw_pos[2] + self.pivot_height * 0.5

        # Check the along-beam offset is within the beam length
        if not (
            -beam_half_length - tolerance
            < local_offset[0]
            < beam_half_length + tolerance
        ):
            return False

        # Check the across-beam offset is within the beam width
        if not abs(local_offset[1]) < beam_half_width + tolerance:
            return False

        # Check Z is above pivot (object is resting on beam)
        if not object_position[2] > min_height:
            return False

        return True

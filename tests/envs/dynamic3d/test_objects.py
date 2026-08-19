"""Tests for primitive object classes (Cube, Cuboid, etc.)."""

import numpy as np
import pytest

from kinder.envs.dynamic3d.objects.base import MujocoObject
from kinder.envs.dynamic3d.objects.generated_objects import GeneratedBowl
from kinder.envs.dynamic3d.objects.primitive_objects import Bin, Cube, Cuboid


def test_cuboid_default_initialization():
    """Test cuboid initialization with default parameters."""
    cuboid = Cuboid("test_cuboid")

    assert cuboid.name == "test_cuboid"
    assert cuboid.joint_name == "test_cuboid_joint"
    assert len(cuboid.size) == 3
    # Default size should be [default_edge_size, default_edge_size, default_edge_size]
    assert all(s == Cuboid.default_edge_size for s in cuboid.size)
    assert cuboid.mass == 0.1


def test_cuboid_with_scalar_size():
    """Test cuboid initialization with scalar size (creates a cube)."""
    cuboid = Cuboid("test_cuboid", options={"size": 0.05})

    assert cuboid.size == [0.05, 0.05, 0.05]


def test_cuboid_with_list_size():
    """Test cuboid initialization with list of dimensions."""
    size = [0.1, 0.2, 0.3]
    cuboid = Cuboid("test_cuboid", options={"size": size})

    assert cuboid.size == size


def test_cuboid_with_invalid_size():
    """Test that invalid size raises ValueError."""
    with pytest.raises(ValueError, match="must be a list of 3 values"):
        Cuboid("test_cuboid", options={"size": [0.1, 0.2]})


def test_cuboid_with_rgba_string():
    """Test cuboid with rgba as string."""
    cuboid = Cuboid("test_cuboid", options={"rgba": "1 0 0 1"})

    assert cuboid.rgba == "1 0 0 1"


def test_cuboid_with_rgba_list():
    """Test cuboid with rgba as list."""
    cuboid = Cuboid("test_cuboid", options={"rgba": [1.0, 0.0, 0.0, 1.0]})

    assert cuboid.rgba == "1.0 0.0 0.0 1.0"


def test_cuboid_with_custom_mass():
    """Test cuboid with custom mass."""
    cuboid = Cuboid("test_cuboid", options={"mass": 0.5})

    assert cuboid.mass == 0.5


def test_cuboid_xml_element_creation():
    """Test that XML element is created correctly."""
    cuboid = Cuboid("test_cuboid", options={"size": [0.1, 0.2, 0.3]})

    assert cuboid.xml_element is not None
    assert cuboid.xml_element.tag == "body"
    assert cuboid.xml_element.get("name") == "test_cuboid"

    # Check for freejoint
    freejoint = cuboid.xml_element.find("freejoint")
    assert freejoint is not None
    assert freejoint.get("name") == "test_cuboid_joint"

    # Check for geom
    geom = cuboid.xml_element.find("geom")
    assert geom is not None
    assert geom.get("type") == "box"
    assert geom.get("size") == "0.1 0.2 0.3"


def test_cuboid_str_repr():
    """Test string representations."""
    cuboid = Cuboid("test_cuboid", options={"size": 0.05, "mass": 0.2})

    str_repr = str(cuboid)
    assert "Cuboid" in str_repr
    assert "test_cuboid" in str_repr

    repr_str = repr(cuboid)
    assert "Cuboid" in repr_str
    assert "test_cuboid" in repr_str
    assert "test_cuboid_joint" in repr_str


def test_cuboid_get_bounding_box_dimensions():
    """Test get_bounding_box_dimensions method."""
    size = [0.1, 0.2, 0.3]
    cuboid = Cuboid("test_cuboid", options={"size": size})

    bb_dims = cuboid.get_bounding_box_dimensions()

    assert bb_dims == (0.2, 0.4, 0.6)  # 2 * each dimension


def test_cuboid_get_bounding_box_from_config_with_scalar():
    """Test get_bounding_box_from_config with scalar size."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = {"size": 0.5}

    bbox = Cuboid.get_bounding_box_from_config(pos, config)

    # Half-extent is 0.5, so bbox should be [pos - 0.5, pos + 0.5] for each dim
    assert bbox == [0.5, 1.5, 2.5, 1.5, 2.5, 3.5]


def test_cuboid_get_bounding_box_from_config_with_list():
    """Test get_bounding_box_from_config with list of dimensions."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = {"size": [0.1, 0.2, 0.3]}

    bbox = Cuboid.get_bounding_box_from_config(pos, config)

    # Expected: [x_min, y_min, z_min, x_max, y_max, z_max]
    assert bbox == [0.9, 1.8, 2.7, 1.1, 2.2, 3.3]


def test_cuboid_get_bounding_box_from_config_default():
    """Test get_bounding_box_from_config with default size."""
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    config = {}

    bbox = Cuboid.get_bounding_box_from_config(pos, config)

    default = Cuboid.default_edge_size
    assert bbox == [-default, -default, -default, default, default, default]


def test_cuboid_get_bounding_box_from_config_invalid_size():
    """Test that invalid size in config raises ValueError."""
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    config = {"size": [0.1, 0.2]}  # Only 2 dimensions

    with pytest.raises(ValueError, match="must be a scalar or list of 3 values"):
        Cuboid.get_bounding_box_from_config(pos, config)


def test_cube_default_initialization():
    """Test cube initialization with default parameters."""
    cube = Cube("test_cube")

    assert cube.name == "test_cube"
    assert cube.joint_name == "test_cube_joint"
    assert len(cube.size) == 3
    assert all(s == Cuboid.default_edge_size for s in cube.size)


def test_cube_with_scalar_size():
    """Test cube initialization with scalar size."""
    cube = Cube("test_cube", options={"size": 0.05})

    assert cube.size == [0.05, 0.05, 0.05]


def test_cube_with_equal_list_size():
    """Test cube initialization with equal dimensions list."""
    cube = Cube("test_cube", options={"size": [0.05, 0.05, 0.05]})

    # Should normalize to scalar internally
    assert cube.size == [0.05, 0.05, 0.05]


def test_cube_inherits_from_cuboid():
    """Test that Cube is a subclass of Cuboid."""
    cube = Cube("test_cube")

    assert isinstance(cube, Cuboid)
    assert isinstance(cube, Cube)


def test_cube_str_repr():
    """Test string representations for Cube."""
    cube = Cube("test_cube", options={"size": 0.05})

    str_repr = str(cube)
    assert "Cube" in str_repr
    assert "test_cube" in str_repr

    repr_str = repr(cube)
    assert "Cube" in repr_str
    assert "test_cube" in repr_str


def test_cube_get_bounding_box_dimensions():
    """Test that cube inherits get_bounding_box_dimensions from Cuboid."""
    cube = Cube("test_cube", options={"size": 0.1})

    bb_dims = cube.get_bounding_box_dimensions()

    # Size is 0.1, so dimensions should be 2 * 0.1 = 0.2 for all axes
    assert bb_dims == (0.2, 0.2, 0.2)


def test_cube_get_bounding_box_from_config():
    """Test that Cube can use Cuboid's get_bounding_box_from_config."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = {"size": 0.5}

    bbox = Cube.get_bounding_box_from_config(pos, config)

    assert bbox == [0.5, 1.5, 2.5, 1.5, 2.5, 3.5]


def test_cube_xml_element_creation():
    """Test that Cube creates valid XML element."""
    cube = Cube("test_cube", options={"size": 0.1})

    assert cube.xml_element is not None
    assert cube.xml_element.tag == "body"
    assert cube.xml_element.get("name") == "test_cube"

    geom = cube.xml_element.find("geom")
    assert geom is not None
    assert geom.get("type") == "box"
    # All dimensions should be equal for a cube
    assert geom.get("size") == "0.1 0.1 0.1"


def test_cuboid_vs_cube_with_same_size():
    """Test that Cuboid and Cube produce same results with equal dimensions."""
    size = 0.1
    cuboid = Cuboid("test_cuboid", options={"size": size})
    cube = Cube("test_cube", options={"size": size})

    assert cuboid.size == cube.size
    assert cuboid.get_bounding_box_dimensions() == cube.get_bounding_box_dimensions()


def test_bounding_box_consistency():
    """Test that bounding box calculations are consistent."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = {"size": 0.5}

    cuboid_bbox = Cuboid.get_bounding_box_from_config(pos, config)
    cube_bbox = Cube.get_bounding_box_from_config(pos, config)

    assert cuboid_bbox == cube_bbox


# Tests for Bin


def test_bin_default_initialization():
    """Test bin initialization with default parameters."""
    bin_obj = Bin("test_bin")

    assert bin_obj.name == "test_bin"
    assert bin_obj.joint_name == "test_bin_joint"
    assert bin_obj.length == 0.1
    assert bin_obj.width == 0.1
    assert bin_obj.height == 0.05
    assert bin_obj.wall_thickness == Bin.default_wall_thickness
    assert bin_obj.mass == 0.1


def test_bin_custom_dimensions():
    """Test bin initialization with custom dimensions."""
    options = {
        "length": 0.2,
        "width": 0.15,
        "height": 0.08,
        "wall_thickness": 0.01,
    }
    bin_obj = Bin("test_bin", options=options)

    assert bin_obj.length == 0.2
    assert bin_obj.width == 0.15
    assert bin_obj.height == 0.08
    assert bin_obj.wall_thickness == 0.01


def test_bin_with_rgba_string():
    """Test bin with rgba as string."""
    bin_obj = Bin("test_bin", options={"rgba": "0.8 0.8 0.8 1"})

    assert bin_obj.rgba == "0.8 0.8 0.8 1"


def test_bin_with_rgba_list():
    """Test bin with rgba as list."""
    bin_obj = Bin("test_bin", options={"rgba": [0.8, 0.8, 0.8, 1.0]})

    assert bin_obj.rgba == "0.8 0.8 0.8 1.0"


def test_bin_with_custom_mass():
    """Test bin with custom mass."""
    bin_obj = Bin("test_bin", options={"mass": 0.2})

    assert bin_obj.mass == 0.2


def test_bin_xml_element_creation():
    """Test that XML element is created correctly with 5 geoms."""
    bin_obj = Bin("test_bin", options={"length": 0.2, "width": 0.15, "height": 0.08})

    assert bin_obj.xml_element is not None
    assert bin_obj.xml_element.tag == "body"
    assert bin_obj.xml_element.get("name") == "test_bin"

    # Check for freejoint
    freejoint = bin_obj.xml_element.find("freejoint")
    assert freejoint is not None
    assert freejoint.get("name") == "test_bin_joint"

    # Check for 5 geoms (1 bottom + 4 walls)
    geoms = bin_obj.xml_element.findall("geom")
    assert len(geoms) == 5

    # All should be box type
    for geom in geoms:
        assert geom.get("type") == "box"


def test_bin_str_repr():
    """Test string representations for Bin."""
    bin_obj = Bin(
        "test_bin",
        options={"length": 0.2, "width": 0.15, "height": 0.08, "mass": 0.2},
    )

    str_repr = str(bin_obj)
    assert "Bin" in str_repr
    assert "test_bin" in str_repr
    assert "0.2" in str_repr  # length
    assert "0.15" in str_repr  # width
    assert "0.08" in str_repr  # height

    repr_str = repr(bin_obj)
    assert "Bin" in repr_str
    assert "test_bin" in repr_str
    assert "test_bin_joint" in repr_str


def test_bin_get_bounding_box_dimensions():
    """Test get_bounding_box_dimensions method for Bin."""
    length = 0.2
    width = 0.15
    height = 0.08
    bin_obj = Bin(
        "test_bin",
        options={"length": length, "width": width, "height": height},
    )

    bb_dims = bin_obj.get_bounding_box_dimensions()

    assert bb_dims == (length, width, height)


def test_bin_get_bounding_box_from_config_default():
    """Test get_bounding_box_from_config with default values."""
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    config = {}

    bbox = Bin.get_bounding_box_from_config(pos, config)

    # Defaults: length=0.1, width=0.1, height=0.05
    # Half-extents: length/2=0.05, width/2=0.05
    # Expected: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Base at (0,0,0), extends to height
    expected = [-0.05, -0.05, 0.0, 0.05, 0.05, 0.05]
    assert bbox == expected


def test_bin_get_bounding_box_from_config_custom():
    """Test get_bounding_box_from_config with custom dimensions."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = {"length": 0.2, "width": 0.15, "height": 0.08}

    bbox = Bin.get_bounding_box_from_config(pos, config)

    # Half-extents: length/2=0.1, width/2=0.075
    # Expected: [x_min, y_min, z_min, x_max, y_max, z_max]
    # x: 1.0 +/- 0.1, y: 2.0 +/- 0.075, z: 3.0 to 3.08
    expected = [0.9, 1.925, 3.0, 1.1, 2.075, 3.08]
    assert bbox == expected


def test_bin_get_bounding_box_from_config_origin_at_base():
    """Test that origin is at base of bin (z_min = pos[2])."""
    pos = np.array([5.0, 6.0, 2.0], dtype=np.float32)
    config = {"length": 0.1, "width": 0.1, "height": 0.05}

    bbox = Bin.get_bounding_box_from_config(pos, config)

    # Check that z_min equals pos[2] (base at ground level)
    assert bbox[2] == pos[2]
    # Check that z_max = z_min + height
    assert bbox[5] == pos[2] + 0.05


def test_bin_geom_positioning():
    """Test that geoms are positioned correctly with base at origin."""
    bin_obj = Bin(
        "test_bin",
        options={"length": 0.2, "width": 0.1, "height": 0.08, "wall_thickness": 0.005},
    )

    geoms = bin_obj.xml_element.findall("geom")
    assert len(geoms) == 5

    # Get positions of geoms
    positions = []
    for geom in geoms:
        pos_str = geom.get("pos")
        if pos_str:
            pos = [float(x) for x in pos_str.split()]
            positions.append(pos)

    # Bottom geom should have z_pos = wall_thickness/2 = 0.0025
    bottom_z = positions[0][2]
    assert abs(bottom_z - 0.0025) < 1e-6

    # Wall geoms should have z_pos >= wall_thickness
    for i in range(1, 5):
        wall_z = positions[i][2]
        assert wall_z >= 0.005


def test_bin_inherits_from_mujoco_object():
    """Test that Bin inherits from MujocoObject."""
    bin_obj = Bin("test_bin")

    assert isinstance(bin_obj, MujocoObject)
    assert isinstance(bin_obj, Bin)


def test_bin_region_with_six_value_range_carries_its_own_z_bounds():
    """A region attached to an object (not the ground) needs real z bounds when it must
    span the object's own height -- e.g. a goal region that has to score anywhere inside a
    bin's interior, not just in a hairline slice at the bin's own local z=0. A 6-value
    range [x_start, y_start, z_start, x_end, y_end, z_end] carries z_start/z_end through
    instead of the historical fixed +/-1cm band (see the four-value test below).
    """
    options = {
        "length": 0.2,
        "width": 0.2,
        "height": 0.1,
        "regions": {
            "goal_region": {
                "target": "test_bin",
                "ranges": [[-0.05, -0.05, 0.0, 0.05, 0.05, 0.08]],
            }
        },
    }
    bin_obj = Bin("test_bin", options=options)

    (region,) = bin_obj.region_objects["goal_region"]
    site = region.site_element
    assert site is not None
    size = [float(v) for v in site.get("size", "").split()]
    pos = [float(v) for v in site.get("pos", "").split()]

    placement_threshold = 0.01
    z_min = pos[2] - size[2]
    z_max = pos[2] + size[2]
    assert z_min == pytest.approx(0.0 - placement_threshold)
    assert z_max == pytest.approx(0.08 + placement_threshold)


def test_bin_region_with_four_value_range_keeps_the_historical_thin_z_band():
    """Same construction, but a 4-value range: the pre-existing behaviour must be
    unchanged by six-value support -- a small, fixed z band around the object's own local
    origin, regardless of the x/y extent requested."""
    options = {
        "length": 0.2,
        "width": 0.2,
        "height": 0.1,
        "regions": {
            "surface_region": {
                "target": "test_bin",
                "ranges": [[-0.05, -0.05, 0.05, 0.05]],
            }
        },
    }
    bin_obj = Bin("test_bin", options=options)

    (region,) = bin_obj.region_objects["surface_region"]
    site = region.site_element
    assert site is not None
    size = [float(v) for v in site.get("size", "").split()]
    pos = [float(v) for v in site.get("pos", "").split()]

    z_min = pos[2] - size[2]
    z_max = pos[2] + size[2]
    assert z_min == pytest.approx(-0.01)
    assert z_max == pytest.approx(0.01)


def test_bin_region_with_bad_range_length_raises():
    """Neither 3 nor 5 values means anything; `_create_regions` should say so rather than
    fail with an opaque unpacking error."""
    options = {
        "regions": {
            "bad_region": {
                "target": "test_bin",
                "ranges": [[0.0, 0.0, 0.0]],
            }
        },
    }
    with pytest.raises(ValueError, match="4 or 6 values"):
        Bin("test_bin", options=options)


# Tests for GeneratedBowl


def test_generated_bowl_default_initialization():
    """Test GeneratedBowl initialization with default parameters."""
    bowl = GeneratedBowl("test_bowl")

    assert bowl.name == "test_bowl"
    assert bowl.joint_name == "test_bowl_joint"
    assert bowl.outer_radius == 0.05
    assert bowl.inner_radius == 0.045
    assert bowl.height == 0.025
    assert bowl.wall_thickness == 0.003
    assert bowl.radial_segments == 32
    assert bowl.vertical_segments == 16
    assert bowl.mass == 0.05


def test_generated_bowl_custom_dimensions():
    """Test GeneratedBowl with custom dimensions."""
    options = {
        "outer_radius": 0.1,
        "inner_radius": 0.09,
        "height": 0.05,
        "wall_thickness": 0.005,
        "radial_segments": 16,
        "vertical_segments": 8,
    }
    bowl = GeneratedBowl("test_bowl", options=options)

    assert bowl.outer_radius == 0.1
    assert bowl.inner_radius == 0.09
    assert bowl.height == 0.05
    assert bowl.wall_thickness == 0.005
    assert bowl.radial_segments == 16
    assert bowl.vertical_segments == 8


def test_generated_bowl_with_rgba_string():
    """Test GeneratedBowl with rgba as string."""
    bowl = GeneratedBowl("test_bowl", options={"rgba": "1 0 0 1"})

    assert bowl.rgba == "1 0 0 1"


def test_generated_bowl_with_rgba_list():
    """Test GeneratedBowl with rgba as list."""
    bowl = GeneratedBowl("test_bowl", options={"rgba": [1.0, 0.0, 0.0, 1.0]})

    assert bowl.rgba == "1.0 0.0 0.0 1.0"


def test_generated_bowl_with_custom_mass():
    """Test GeneratedBowl with custom mass."""
    bowl = GeneratedBowl("test_bowl", options={"mass": 0.2})

    assert bowl.mass == 0.2


def test_generated_bowl_mesh_generation():
    """Test that mesh file is generated and has valid path."""
    bowl = GeneratedBowl("test_bowl")

    assert bowl.mesh_file is not None
    assert isinstance(bowl.mesh_file, str)
    assert bowl.mesh_file.endswith(".obj")
    assert bowl.mesh_name == "test_bowl_bowl_mesh"


def test_generated_bowl_xml_element_creation():
    """Test that XML element is created correctly."""
    bowl = GeneratedBowl("test_bowl")

    assert bowl.xml_element is not None
    assert bowl.xml_element.tag == "body"
    assert bowl.xml_element.get("name") == "test_bowl"

    # Check for freejoint
    freejoint = bowl.xml_element.find("freejoint")
    assert freejoint is not None
    assert freejoint.get("name") == "test_bowl_joint"

    # Check for geom
    geom = bowl.xml_element.find("geom")
    assert geom is not None
    assert geom.get("type") == "mesh"
    assert geom.get("mesh") == "test_bowl_bowl_mesh"


def test_generated_bowl_get_assets():
    """Test that get_assets returns mesh asset element."""
    bowl = GeneratedBowl("test_bowl")

    assets = bowl.get_assets()

    assert len(assets) == 1
    mesh_elem = assets[0]
    assert mesh_elem.tag == "mesh"
    assert mesh_elem.get("file") == bowl.mesh_file
    assert mesh_elem.get("name") == "test_bowl_bowl_mesh"


def test_generated_bowl_str_repr():
    """Test string representations."""
    bowl = GeneratedBowl("test_bowl", options={"outer_radius": 0.1, "mass": 0.2})

    str_repr = str(bowl)
    assert "GeneratedBowl" in str_repr
    assert "test_bowl" in str_repr

    repr_str = repr(bowl)
    assert "GeneratedBowl" in repr_str
    assert "test_bowl" in repr_str
    assert "test_bowl_joint" in repr_str


def test_generated_bowl_get_bounding_box_dimensions():
    """Test get_bounding_box_dimensions method."""
    outer_radius = 0.1
    height = 0.05
    bowl = GeneratedBowl(
        "test_bowl", options={"outer_radius": outer_radius, "height": height}
    )

    bb_dims = bowl.get_bounding_box_dimensions()

    assert bb_dims == (2 * outer_radius, 2 * outer_radius, height)


def test_generated_bowl_get_bounding_box_from_config():
    """Test get_bounding_box_from_config static method."""
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = {"outer_radius": 0.05, "height": 0.025}

    bbox = GeneratedBowl.get_bounding_box_from_config(pos, config)

    # Expected: [x_min, y_min, z_min, x_max, y_max, z_max]
    # x: 1.0 +/- 0.05, y: 2.0 +/- 0.05, z: 3.0 +/- 0.0125
    expected = [
        0.95,  # x_min
        1.95,  # y_min
        2.9875,  # z_min
        1.05,  # x_max
        2.05,  # y_max
        3.0125,  # z_max
    ]
    assert bbox == expected


def test_generated_bowl_get_bounding_box_from_config_defaults():
    """Test get_bounding_box_from_config with default values."""
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    config = {}

    bbox = GeneratedBowl.get_bounding_box_from_config(pos, config)

    # Defaults: outer_radius=0.05, height=0.025
    expected = [
        -0.05,  # x_min
        -0.05,  # y_min
        -0.0125,  # z_min
        0.05,  # x_max
        0.05,  # y_max
        0.0125,  # z_max
    ]
    assert bbox == expected


def test_generated_bowl_get_bounding_box_from_config_custom():
    """Test get_bounding_box_from_config with custom values."""
    pos = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    config = {"outer_radius": 0.2, "height": 0.1}

    bbox = GeneratedBowl.get_bounding_box_from_config(pos, config)

    # Expected: [x_min, y_min, z_min, x_max, y_max, z_max]
    # x: 5.0 +/- 0.2, y: 6.0 +/- 0.2, z: 7.0 +/- 0.05
    expected = [
        4.8,  # x_min
        5.8,  # y_min
        6.95,  # z_min
        5.2,  # x_max
        6.2,  # y_max
        7.05,  # z_max
    ]
    assert bbox == expected


def test_generated_bowl_mesh_generation_creates_vertices_and_faces():
    """Test that mesh generation creates valid vertices and faces."""
    bowl = GeneratedBowl(
        "test_bowl", options={"radial_segments": 8, "vertical_segments": 4}
    )

    # pylint: disable=protected-access
    vertices, faces = bowl._generate_bowl_mesh()

    # Check that we have vertices and faces
    assert vertices.shape[0] > 0
    assert vertices.shape[1] == 3
    assert faces.shape[0] > 0
    assert faces.shape[1] == 3

    # Check that all face indices are valid
    max_vertex_index = vertices.shape[0] - 1
    assert np.all(faces >= 0)
    assert np.all(faces <= max_vertex_index)


# Tests for check_in_region method


class MockMujocoEnv:
    """Mock environment for testing MujocoObject methods."""

    def __init__(self, joint_pos: list[float], joint_quat: list[float] | None = None):
        """Initialize mock environment with fixed joint position."""
        self.joint_pos = np.array(joint_pos, dtype=np.float32)
        self.joint_quat = (
            np.array(joint_quat, dtype=np.float32)
            if joint_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

    def get_joint_pos_quat(self, _joint_name: str):
        """Return stored position and quaternion."""
        return self.joint_pos, self.joint_quat

    def set_joint_pos_quat(self, _joint_name: str, pos, quat):
        """Update stored position and quaternion."""
        self.joint_pos = np.array(pos, dtype=np.float32)
        self.joint_quat = np.array(quat, dtype=np.float32)

    def set_joint_vel(self, _joint_name: str, linear_vel, angular_vel):
        """Placeholder for velocity setting."""

    def get_joint_vel(self, _joint_name: str):
        """Return zero velocities."""
        return (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

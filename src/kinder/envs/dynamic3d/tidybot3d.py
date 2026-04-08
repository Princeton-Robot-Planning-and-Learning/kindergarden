"""TidyBot 3D environment wrapper for KinDER."""

import abc
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray
from relational_structs import Array, Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta, KinDEREnvConfig
from kinder.envs.dynamic3d.base_env import (
    ObjectCentricDynamic3DRobotEnv,
)
from kinder.envs.dynamic3d.object_types import (
    MujocoObjectTypeFeatures,
    MujocoRBY1ARobotObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.objects import (
    MujocoFixture,
    MujocoGround,
    MujocoObject,
    get_fixture_class,
    get_object_class,
)
from kinder.envs.dynamic3d.objects.generated_objects import GeneratedSeesaw
from kinder.envs.dynamic3d.placement_samplers import (
    sample_collision_free_positions,
)
from kinder.envs.dynamic3d.robots import (
    RBY1ARobotActionSpace,
    RBY1ARobotEnv,
    TidyBot3DRobotActionSpace,
    TidyBotRobotEnv,
)
from kinder.envs.dynamic3d.scene_loader import SceneLoader
from kinder.envs.dynamic3d.tidybot_rewards import create_reward_calculator
from kinder.envs.dynamic3d.utils import (
    compute_camera_euler,
    convert_yaw_to_quaternion,
)


@dataclass(frozen=True)
class TidyBot3DConfig(KinDEREnvConfig, metaclass=FinalConfigMeta):
    """Configuration for TidyBot3D environment."""

    control_frequency: int = 10
    horizon: int = 1000
    camera_names: list[str] = field(default_factory=lambda: ["overview"])
    camera_width: int = 640
    camera_height: int = 480
    show_viewer: bool = False
    act_delta: bool = True


class ObjectCentricRobotEnv(ObjectCentricDynamic3DRobotEnv[TidyBot3DConfig]):
    """TidyBot 3D environment with mobile manipulation tasks."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: TidyBot3DConfig = TidyBot3DConfig(),
        seed: int | None = None,
        scene_type: str = "ground",
        num_objects: int = 3,
        task_config_path: str | None = None,
        show_images: bool = False,
        scene_bg: bool | str | None = None,
        scene_render_camera: str | None = None,
        **kwargs,
    ) -> None:
        # Initialize ObjectCentricKinDEREnv first
        super().__init__(config, **kwargs)

        # Store instance attributes from kwargs
        self.scene_type = scene_type
        self.num_objects = num_objects
        self.show_images = show_images
        self.seed = seed
        self.config = config

        # Parse task configuration
        if task_config_path is None:
            # Default task config based on scene_type and num_objects
            task_config_path = (
                f"./tasks/tidybot-{self.scene_type}-o{self.num_objects}.json"
            )
        if not os.path.isabs(task_config_path):
            task_config_path = str(Path(__file__).parent / task_config_path)
        assert os.path.exists(
            task_config_path
        ), f"task_config_path {task_config_path} does not exist."
        self.task_config_path = task_config_path
        with open(task_config_path, "r", encoding="utf-8") as f:
            self.task_config = json.load(f)

        # Check if scene key is a string (scene reference) and load the scene config
        scene_value = self.task_config.get("scene")
        if isinstance(scene_value, str):
            scene_path = str(Path(__file__).parent / "scenes" / f"{scene_value}.json")
            assert os.path.exists(
                scene_path
            ), f"Scene config file {scene_path} does not exist."
            with open(scene_path, "r", encoding="utf-8") as f:
                scene_config = json.load(f)
            # Merge scene_config into task_config recursively for nested dicts
            self._merge_configs(self.task_config, scene_config)

        # Apply scene configuration based on scene_bg parameter
        # scene_bg can be:
        #   - True: Use the mimiclabs scene defined in task config
        #   - False/None: Use "simple" scene
        #   - str: Use the explicit scene name (for backwards compatibility)
        self._scene_bg = scene_bg
        self._apply_scene_bg(scene_bg)

        # Set camera names from config
        self.camera_names = config.camera_names.copy()

        # Update camera names based on scene type
        scene_config = self.task_config.get("_active_scene", {})
        if scene_config.get("type") == "mimiclabs":
            # MimicLabs scenes define: frontview, birdview, agentview, sideview
            self.camera_names = ["frontview", "birdview", "agentview", "sideview"]

        if "cameras" in self.task_config:
            self.camera_names.extend(list(self.task_config["cameras"].keys()))

        # Initialize robot environment
        self.robot_type = list(self.task_config["robots"].keys())[0]
        self.robot_name = list(self.task_config["robots"][self.robot_type].keys())[0]
        robot_cls = {"tidybot": TidyBotRobotEnv, "rby1a": RBY1ARobotEnv}[
            self.robot_type
        ]
        self._robot_env = robot_cls(
            name=self.robot_name,
            control_frequency=self.config.control_frequency,
            act_delta=self.config.act_delta,
            horizon=self.config.horizon,
            camera_names=self.camera_names,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            seed=seed if seed is not None else self.seed,
            show_viewer=self.config.show_viewer,
        )

        # Update camera names since robot may have added its own cameras.
        self.camera_names = self._robot_env.camera_names.copy()

        # This camera's render will be returned by default.
        self._render_camera_name: str | None = scene_render_camera
        assert (
            self._render_camera_name is None
            or self._render_camera_name in self.camera_names
        ), (
            f"Render camera '{self._render_camera_name}' not in available "
            f"cameras: {self.camera_names}"
        )

        # Initialize empty object and fixture lists, and ground fixture.
        # These will be populated based on the task configuration
        # in the _create_scene_xml() method.
        self._objects: list[MujocoObject] = []
        self._objects_dict: dict[str, MujocoObject] = {}
        self._fixtures_dict: dict[str, MujocoFixture] = {}
        self._ground_fixture: MujocoGround | None = None

        self._reward_calculator = create_reward_calculator(scene_type, num_objects)

        # Store current state
        self._current_state: ObjectCentricState | None = None

    def _merge_configs(
        self, base_config: dict[str, Any], update_config: dict[str, Any]
    ) -> None:
        """Recursively merge update_config into base_config.

        For nested dictionaries (like "fixtures", "objects", "regions"), merge the
        contents rather than replacing. For lists, append values from update_config
        to base_config. For other values, the update_config value takes precedence.

        Args:
            base_config: Dictionary to merge into (modified in-place)
            update_config: Dictionary to merge from

        Raises:
            AssertionError: If "goal_state" is present in update_config
        """
        assert "goal_state" not in update_config, (
            "Merging goal_state from scene config is not supported. "
            "goal_state should only be defined in the task config."
        )

        for key, update_value in update_config.items():
            if key in base_config:
                if isinstance(base_config[key], dict) and isinstance(
                    update_value, dict
                ):
                    # Both are dicts - recursively merge them
                    self._merge_configs(base_config[key], update_value)
                elif isinstance(base_config[key], list) and isinstance(
                    update_value, list
                ):
                    # Both are lists - append update values to base
                    base_config[key].extend(update_value)
                else:
                    # Otherwise, use the update value
                    base_config[key] = update_value
            else:
                # Key not in base_config, add it
                base_config[key] = update_value

    def _apply_scene_bg(self, scene_bg: bool | str | None) -> None:
        """Apply scene background configuration based on scene_bg parameter.

        Looks up the scene configuration (including position) from the task JSON's
        scene dict, and stores it in task_config["_active_scene"] for use by
        the scene loader.

        Args:
            scene_bg: Scene background setting. Supports:
                - True: Use the mimiclabs scene defined in task config
                - False/None: Use "simple" scene
                - "simple": Use default ground scene
                - "mimiclabs-labN": Use MimicLabs labN scene (N=2-8)
        """
        # Get scene configs from task JSON (format: dict of scene_name -> config)
        scene_configs = self.task_config.get("scene", {})

        # Convert bool/None to scene name
        if scene_bg is None or scene_bg is False:
            scene_bg_name = "simple"
        elif scene_bg is True:
            # Find the mimiclabs scene in task config
            mimiclabs_scenes = [k for k in scene_configs if k.startswith("mimiclabs-")]
            if not mimiclabs_scenes:
                raise ValueError(
                    "scene_bg=True but no mimiclabs scene found in task config. "
                    f"Available scenes: {list(scene_configs.keys())}"
                )
            scene_bg_name = mimiclabs_scenes[0]  # Use the first (and only) mimiclabs
        else:
            scene_bg_name = scene_bg  # It's already a string

        # Look up the scene config for the requested scene_bg
        if scene_bg_name not in scene_configs:
            raise ValueError(
                f"Scene '{scene_bg_name}' not found in task config. "
                f"Available scenes: {list(scene_configs.keys())}"
            )

        scene_config = scene_configs[scene_bg_name]
        position = scene_config.get("position", [0, 0, 0])

        # Build the active scene config
        if scene_bg_name == "simple":
            self.task_config["_active_scene"] = {
                "type": "simple",
                "position": position,
            }
        elif scene_bg_name.startswith("mimiclabs-lab"):
            # Extract lab number from scene_bg_name (e.g., "mimiclabs-lab2" -> 2)
            lab_str = scene_bg_name.split("-lab")[-1]
            try:
                lab_num = int(lab_str)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse lab number from {scene_bg_name}. "
                    f"Expected format: 'mimiclabs-lab2' through 'mimiclabs-lab8'"
                ) from e
            if not 2 <= lab_num <= 8:
                raise ValueError(f"MimicLabs lab number must be 2-8, got {lab_num}")
            self.task_config["_active_scene"] = {
                "type": "mimiclabs",
                "lab": lab_num,
                "position": position,
                "wall_collision": scene_config.get("wall_collision", False),
            }
        else:
            raise ValueError(
                f"Unknown scene_bg: {scene_bg_name}. "
                f"Supported values: 'simple', 'mimiclabs-lab2' through 'mimiclabs-lab8'"
            )

    def _vectorize_observation(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        """Convert TidyBot observation dict to vector."""
        obs_vector: list[float] = []
        for key in sorted(obs.keys()):  # Sort for consistency
            value = obs[key]
            obs_vector.extend(value.flatten())
        return np.array(obs_vector, dtype=np.float32)

    def _setup_cameras(self, root: ET.Element) -> None:
        """Setup cameras from task configuration.

        Reads camera configurations from self.task_config["cameras"] and creates
        corresponding camera XML elements in the scene.

        Expected camera config kwargs:
            position: [x, y, z] camera position (default: [0, 0, 1])
            lookat: [x, y, z] point camera looks at (default: [0, 0, 0])
            fovy: field of view angle in degrees (default: 45)
            resolution: [width, height] in pixels (default: [640, 480])

        Args:
            root: Root element of the MuJoCo XML tree
        """
        if "cameras" not in self.task_config:
            return

        cameras_config = self.task_config["cameras"]
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("No worldbody found in XML; cannot add cameras.")

        for camera_name, camera_config in cameras_config.items():
            position = camera_config.get("position", [0, 0, 1])
            lookat = camera_config.get("lookat", [0, 0, 0])
            fovy = camera_config.get("fovy", 45)
            resolution = camera_config.get("resolution", [640, 480])

            # Validate parameters
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise ValueError(
                    f"Camera '{camera_name}': position must be a 3-element list, "
                    f"got {position}"
                )
            if not isinstance(lookat, (list, tuple)) or len(lookat) != 3:
                raise ValueError(
                    f"Camera '{camera_name}': lookat must be a 3-element list, "
                    f"got {lookat}"
                )
            if not isinstance(fovy, (int, float)) or fovy <= 0:
                raise ValueError(
                    f"Camera '{camera_name}': fovy must be a positive number, "
                    f"got {fovy}"
                )
            if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
                raise ValueError(
                    f"Camera '{camera_name}': resolution must be a 2-element list, "
                    f"got {resolution}"
                )
            if not all(isinstance(r, int) and r > 0 for r in resolution):
                raise ValueError(
                    f"Camera '{camera_name}': resolution must contain positive "
                    f"integers, got {resolution}"
                )

            # Cast to list[float] after validation
            position_list: list[float] = list(position)  # type: ignore[arg-type]
            lookat_list: list[float] = list(lookat)  # type: ignore[arg-type]

            # Compute euler angles from position and lookat
            euler = compute_camera_euler(position_list, lookat_list)

            # Create camera element
            camera_elem = ET.SubElement(worldbody, "camera")
            camera_elem.set("name", camera_name)
            camera_elem.set("pos", f"{position[0]} {position[1]} {position[2]}")
            camera_elem.set("euler", f"{euler[0]} {euler[1]} {euler[2]}")
            camera_elem.set("fovy", str(fovy))
            camera_elem.set("resolution", f"{resolution[0]} {resolution[1]}")

    def _create_scene_xml(self) -> str:
        """Create the MuJoCo XML string for the current scene configuration."""

        # Set model path to local models directory
        model_base_path = Path(__file__).parent / "models" / "stanford_tidybot"

        # Load scene XML using SceneLoader
        # Use _active_scene which is set by _apply_scene_bg() based on scene_bg param
        scene_config = self.task_config.get("_active_scene", {"type": "simple"})
        xml_string = SceneLoader.load_scene(scene_config, model_base_path)

        # Insert objects in scene
        root = ET.fromstring(xml_string)
        # Get or create asset section for adding meshes/textures/materials
        asset_section = root.find("asset")
        if asset_section is None:
            asset_section = ET.Element("asset")
            # Insert asset section after visual section if it exists
            visual_section = root.find("visual")
            if visual_section is not None:
                visual_index = list(root).index(visual_section)
                root.insert(visual_index + 1, asset_section)
            else:
                root.insert(0, asset_section)
        worldbody = root.find("worldbody")
        if worldbody is not None:
            if self.task_config is not None:
                all_fixtures = self.task_config.get("fixtures", {})
                fixtures: dict[str, dict[str, dict[str, Any]]] = {}

                # Find regions on ground and create ground fixture
                regions_on_ground = {}
                all_regions = self.task_config.get("regions", {})
                for region_name, region_config in all_regions.items():
                    if region_config["target"] == "ground":
                        regions_on_ground[region_name] = region_config
                # Create ground fixture for region sampling
                self._ground_fixture = MujocoGround(
                    regions=regions_on_ground,
                    worldbody=worldbody,
                )
                self._ground_fixture.visualize_regions()

                # Create fixture region names and pos/yaw samplers dicts
                entity_region_names: dict[str, str] = {}
                entity_pos_yaw_samplers: dict[str, Any] = {}

                # Go through initial_state predicates and find fixtures that
                # need to be placed, and create pos/yaw samplers for them
                init_predicates = self.task_config.get("initial_state", [])
                for pred in init_predicates:
                    if pred[0] in ["on", "in"] and len(pred) == 3:
                        fixture_name = pred[1]
                        region_name = pred[2]

                        # Check if this fixture exists in any fixture type and add to
                        # fixtures dict
                        fixture_found = False
                        for fixture_type, fixture_configs in all_fixtures.items():
                            if fixture_name in fixture_configs:
                                fixture_found = True
                                # Add this fixture type and fixture to filtered
                                # fixtures dict
                                if fixture_type not in fixtures:
                                    fixtures[fixture_type] = {}
                                fixtures[fixture_type][fixture_name] = fixture_configs[
                                    fixture_name
                                ]
                                break

                        if fixture_found:
                            if pred[0] == "in":
                                warning_msg = (
                                    "Warning: Found 'in' predicate for fixture "
                                    "placement, which is not supported. "
                                    "Falling back to 'on' predicate."
                                )
                                print(warning_msg)
                            region_config = self.task_config["regions"][region_name]
                            # Assert that the region target is ground
                            assert region_config["target"] == "ground", (
                                f"Region {region_name} for fixture {fixture_name} "
                                "must have target 'ground', got "
                                f"'{region_config['target']}'"
                            )

                            # Add pos/yaw samplers on ground for this fixture
                            entity_region_names[fixture_name] = region_name
                            entity_pos_yaw_samplers[fixture_name] = (
                                self._ground_fixture.sample_pose_in_region
                            )

                # Sample collision-free positions for fixtures
                # Note: we do not pass entity_check_in_region, and so only the
                # center of the fixture is guaranteed to be within the region
                fixture_poses = sample_collision_free_positions(
                    fixtures,
                    self.np_random,
                    entity_region_names=entity_region_names,
                    entity_pos_yaw_samplers=entity_pos_yaw_samplers,
                )

                # Insert filtered fixtures
                for fixture_type, fixture_configs in fixtures.items():

                    for fixture_name, fixture_config in fixture_configs.items():
                        # Sample collision-free position for the fixture
                        fixture_pose = fixture_poses[fixture_type][fixture_name]
                        fixture_pos = fixture_pose["position"]
                        fixture_yaw = fixture_pose["yaw"]

                        # Find regions for this fixture if specified
                        regions_in_fixture = {}
                        all_regions = self.task_config.get("regions", {})
                        for region_name, region_config in all_regions.items():
                            if region_config["target"] == fixture_name:
                                regions_in_fixture[region_name] = region_config

                        # Create new fixture with configuration dictionary
                        fixture_cls = get_fixture_class(fixture_type)
                        new_fixture = fixture_cls(
                            name=fixture_name,
                            fixture_config=fixture_config,
                            position=fixture_pos,
                            yaw=fixture_yaw,
                            regions=regions_in_fixture,
                            env=self._robot_env,
                        )
                        new_fixture.visualize_regions()
                        self._fixtures_dict[fixture_name] = new_fixture
                        fixture_body = new_fixture.xml_element
                        worldbody.append(fixture_body)

                # Insert all objects
                objects = self.task_config.get("objects", {})
                for object_type, object_configs in objects.items():
                    for object_name, object_config in object_configs.items():
                        # Find regions for this object if specified
                        regions_in_object = {}
                        all_regions = self.task_config.get("regions", {})
                        for region_name, region_config in all_regions.items():
                            if region_config["target"] == object_name:
                                regions_in_object[region_name] = region_config

                        # Add regions to object config
                        obj_options = object_config.copy() if object_config else {}
                        obj_options["regions"] = regions_in_object

                        obj_cls = get_object_class(object_type)
                        obj = obj_cls(
                            name=object_name,
                            env=self._robot_env,
                            options=obj_options,
                        )
                        obj.visualize_regions()
                        body = obj.xml_element
                        worldbody.append(body)
                        self._objects.append(obj)
                        self._objects_dict[object_name] = obj

                        # Add assets if the object has them
                        # (e.g., RoboCasa, GeneratedBowl)
                        if hasattr(obj, "get_assets"):
                            obj_assets = obj.get_assets()
                            # Get existing asset names to avoid duplicates
                            # Track per asset type since MuJoCo allows same name
                            # for different types
                            existing_names: dict[str, set[str]] = {}
                            for existing_asset in asset_section:
                                asset_tag = existing_asset.tag
                                asset_name = existing_asset.get("name")
                                if asset_name:
                                    if asset_tag not in existing_names:
                                        existing_names[asset_tag] = set()
                                    existing_names[asset_tag].add(asset_name)

                            # Add all mesh, texture, and material elements
                            # to asset section, skipping duplicates within same type
                            for asset_elem in obj_assets:
                                asset_tag = asset_elem.tag
                                asset_name = asset_elem.get("name")
                                if (
                                    asset_name
                                    and asset_tag in existing_names
                                    and asset_name in existing_names[asset_tag]
                                ):
                                    continue
                                asset_section.append(asset_elem)
                                if asset_name:
                                    if asset_tag not in existing_names:
                                        existing_names[asset_tag] = set()
                                    existing_names[asset_tag].add(asset_name)

            # Setup cameras from task configuration
            self._setup_cameras(root)

            # Get XML string from tree
            xml_string = ET.tostring(root, encoding="unicode")

        return xml_string

    def _initialize_object_poses(self) -> None:
        """Initialize object poses in the environment."""

        assert self._robot_env is not None, "Robot environment not initialized"
        assert self._robot_env.sim is not None, "Simulation not initialized"

        # Collect all objects and their target regions
        init_predicates = self.task_config.get("initial_state", [])

        # Separate objects by their target (ground or fixture)
        ground_objects: dict[str, dict[str, Any]] = {}
        # obj_name -> (target, region, pred_type)
        fixture_objects: dict[str, tuple[str, str, str]] = {}

        # Go through all initial state predicates and categorize objects
        for pred in init_predicates:
            if pred[0] in ["on", "in"]:
                pred_type = pred[0]
                obj_name = pred[1]
                region_name = pred[2]

                # Skip fixtures, they are static
                if obj_name in self._fixtures_dict:
                    continue

                if obj_name not in self._objects_dict:
                    if obj_name == self.robot_name:
                        continue
                    raise ValueError(f"Object {obj_name} not found in environment.")

                region_config = self.task_config["regions"][region_name]
                target = region_config["target"]

                if target == "ground":
                    # Collect ground-placed objects
                    obj = self._objects_dict[obj_name]
                    # pylint: disable=no-member
                    obj_type = (
                        obj.__class__.REGISTERED_NAME  # type: ignore[attr-defined]
                    )
                    obj_config = self.task_config["objects"][obj_type].get(obj_name, {})
                    ground_objects[obj_name] = obj_config
                    fixture_objects[obj_name] = (target, region_name, pred_type)
                else:
                    # Handle fixture-placed objects separately below
                    fixture_objects[obj_name] = (target, region_name, pred_type)

        # Place objects on ground with collision checking
        if ground_objects:
            # Reuse the ground fixture already created in _create_scene_xml()
            assert self._ground_fixture is not None, "Ground fixture not initialized"
            ground_fixture = self._ground_fixture

            # Prepare entity dicts for sample_collision_free_positions
            entity_region_names: dict[str, str] = {}
            entity_pos_yaw_samplers: dict[str, Any] = {}
            entity_check_in_region: dict[str, Any] = {}

            # Process regions and samplers on ground
            ground_object_configs: dict[str, dict[str, Any]] = {}
            for obj_name, region_info in fixture_objects.items():
                target, region_name, pred_type = region_info
                if target == "ground" and obj_name in ground_objects:
                    entity_region_names[obj_name] = region_name
                    entity_pos_yaw_samplers[obj_name] = (
                        ground_fixture.sample_pose_in_region
                    )
                    if pred_type == "in":
                        # If object needs to be "in" the region, we additionally check
                        # for the object bbox to lie entirely within the region
                        entity_check_in_region[obj_name] = (
                            ground_fixture.check_in_region
                        )
                    # Get the object type for this object
                    obj = self._objects_dict[obj_name]
                    # pylint: disable=no-member
                    obj_type = (
                        obj.__class__.REGISTERED_NAME  # type: ignore[attr-defined]
                    )
                    if obj_type not in ground_object_configs:
                        ground_object_configs[obj_type] = {}
                    ground_object_configs[obj_type][obj_name] = ground_objects[obj_name]

            # Sample collision-free positions for ground objects
            object_poses = sample_collision_free_positions(
                ground_object_configs,
                self.np_random,
                entity_region_names=entity_region_names,
                entity_pos_yaw_samplers=entity_pos_yaw_samplers,
                entity_check_in_region=entity_check_in_region,
            )

            # Set poses for ground-placed objects
            for obj_type, obj_poses_dict in object_poses.items():
                for obj_name in obj_poses_dict:
                    pos = obj_poses_dict[obj_name]["position"]
                    yaw = obj_poses_dict[obj_name]["yaw"]

                    obj = self._objects_dict[obj_name]
                    quat = convert_yaw_to_quaternion(yaw)
                    obj.set_pose(pos, quat)

        # Place objects on fixtures
        fixture_entity_region_names: dict[str, str] = {}
        fixture_entity_pos_yaw_samplers: dict[str, Any] = {}
        fixture_entity_check_in_region: dict[str, Any] = {}
        fixture_object_configs: dict[str, dict[str, Any]] = {}

        for obj_name, region_info in fixture_objects.items():
            target, region_name, pred_type = region_info

            if target == "ground":
                continue  # Already handled above

            # Get target fixture
            target_fixture = self._fixtures_dict[target]

            # Add to entity dicts
            fixture_entity_region_names[obj_name] = region_name
            fixture_entity_pos_yaw_samplers[obj_name] = (
                target_fixture.sample_pose_in_region
            )
            if pred_type == "in":
                # If object needs to be "in" the region, we additionally check
                # for the object bbox to lie entirely within the region
                fixture_entity_check_in_region[obj_name] = (
                    target_fixture.check_in_region
                )

            # Get the object type for this object
            obj = self._objects_dict[obj_name]
            obj_type = obj.__class__.REGISTERED_NAME  # type: ignore[attr-defined]
            if obj_type not in fixture_object_configs:
                fixture_object_configs[obj_type] = {}
            obj_config_dict = self.task_config.get("objects", {})
            fixture_object_configs[obj_type][obj_name] = obj_config_dict.get(
                obj_type, {}
            ).get(obj_name, {})

        # Sample collision-free positions for all fixture-placed objects
        if (
            fixture_entity_region_names
        ):  # Only sample if there are fixture-placed objects
            object_poses = sample_collision_free_positions(
                fixture_object_configs,
                self.np_random,
                entity_region_names=fixture_entity_region_names,
                entity_pos_yaw_samplers=fixture_entity_pos_yaw_samplers,
                entity_check_in_region=fixture_entity_check_in_region,
            )

            # Set poses for fixture-placed objects
            for obj_type, obj_poses_dict in object_poses.items():
                for obj_name in obj_poses_dict:
                    pos = obj_poses_dict[obj_name]["position"]
                    yaw = obj_poses_dict[obj_name]["yaw"]

                    obj = self._objects_dict[obj_name]
                    quat = convert_yaw_to_quaternion(yaw)
                    obj.set_pose(pos, quat)

        self._robot_env.sim.forward()

    @abc.abstractmethod
    def _create_action_space(  # type: ignore
        self, config: TidyBot3DConfig
    ) -> Space[Array]:
        """Create action space for TidyBot's control interface."""

    def _initialize_robot_pose(self) -> None:
        """Initialize the robot in the environment."""

        # Go through predicates, find the ones that specify the robot's initial pose
        init_predicates = self.task_config.get("initial_state", [])
        robot_predicates = []
        for pred in init_predicates:
            if (
                len(pred) >= 3
                and pred[0] in ["on", "in"]
                and pred[1] == self.robot_name
            ):
                robot_predicates.append(pred)
                if pred[0] == "in":
                    print(
                        "Warning: Found 'in' predicate for robot initial pose, "
                        "which is not supported. Falling back to 'on' predicate."
                    )

        # Assert there is exactly one predicate for the robot
        assert len(robot_predicates) <= 1, (
            f"Expected at most 1 predicate for robot '{self.robot_name}', "
            f"got {len(robot_predicates)}"
        )

        if not robot_predicates:
            # Define limits for x, y, and yaw
            x_limit = (-1.0, 1.0)
            y_limit = (-1.0, 1.0)
            yaw_limit = (-np.pi, np.pi)
            # Sample random values within the limits
            x = self.np_random.uniform(*x_limit)
            y = self.np_random.uniform(*y_limit)
            yaw = self.np_random.uniform(*yaw_limit)
        else:
            # Extract region name
            region_name = robot_predicates[0][2]
            region_config = self.task_config["regions"][region_name]

            # Assert that the region target is ground
            assert region_config["target"] == "ground", (
                f"Region '{region_name}' for robot must have target 'ground', "
                f"got '{region_config['target']}'"
            )

            # Sample pose in region using ground fixture
            assert self._ground_fixture is not None, "Ground fixture not initialized"
            x, y, _, yaw = self._ground_fixture.sample_pose_in_region(
                region_name, self.np_random
            )

        # Set robot base position and yaw orientation
        self._robot_env.set_robot_base_pos_yaw(x, y, yaw)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        """Reset the environment and return object-centric observation."""

        # Reset the random seed
        self._robot_env.seed(seed=seed)
        self.np_random = self._robot_env.np_random

        # Create scene XML
        self._objects = []
        self._objects_dict = {}
        self._fixtures_dict = {}
        xml_string = self._create_scene_xml()

        # Reset the underlying TidyBot robot environment
        robot_options = options.copy() if options is not None else {}
        robot_options["xml"] = xml_string
        self._robot_env.reset(options=robot_options)

        # Initialize object poses
        self._initialize_object_poses()

        # Initialize the robot pose
        self._initialize_robot_pose()

        # Get object-centric observation
        self._current_state = self._get_object_centric_state()

        return self._get_current_state(), {}

    def reset_with_images(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any], dict[str, Any]]:
        """Reset the environment and return object-centric observation."""

        # Reset the random seed
        self._robot_env.seed(seed=seed)
        self.np_random = self._robot_env.np_random

        # Create scene XML
        self._objects = []
        self._objects_dict = {}
        self._fixtures_dict = {}
        xml_string = self._create_scene_xml()

        # Reset the underlying TidyBot robot environment
        robot_options = options.copy() if options is not None else {}
        robot_options["xml"] = xml_string
        self._robot_env.reset(options=robot_options)

        # Initialize object poses
        self._initialize_object_poses()

        # Get object-centric observation
        self._current_state = self._get_object_centric_state()

        return self._get_current_state(), {}, self._get_all_obs()

    def _get_state(self) -> ObjectCentricState:
        assert self._current_state is not None, "Need to call reset() first"
        return self._current_state.copy()

    def _set_state(self, state: ObjectCentricState) -> None:
        """Set the environment to the given state."""
        # Reset the robot.
        self._set_robot_state(state)

        # Reset the objects.
        for mujoco_object in self._objects:
            obj = state.get_object_from_name(mujoco_object.name)
            position = [state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z")]
            orientation = [
                state.get(obj, "qw"),
                state.get(obj, "qx"),
                state.get(obj, "qy"),
                state.get(obj, "qz"),
            ]
            mujoco_object.set_pose(position, orientation)
            linear_velocity = [
                state.get(obj, "vx"),
                state.get(obj, "vy"),
                state.get(obj, "vz"),
            ]
            angular_velocity = [
                state.get(obj, "wx"),
                state.get(obj, "wy"),
                state.get(obj, "wz"),
            ]
            mujoco_object.set_velocity(linear_velocity, angular_velocity)
        # NOTE: Fixtures are static (without joints), so we cannot set their state.

        assert self._robot_env is not None, "Robot environment not initialized"
        assert self._robot_env.sim is not None, "Simulation not initialized"
        self._robot_env.sim.forward()

        # Update the cached current state
        self._current_state = self._get_object_centric_state()

    def _visualize_image_in_window(
        self, image: NDArray[np.uint8], window_name: str
    ) -> None:
        """Visualize an image in an OpenCV window."""
        if image.dtype == np.uint8 and len(image.shape) == 3:
            # Convert RGB to BGR for proper color display in OpenCV
            display_image = cv.cvtColor(  # pylint: disable=no-member
                image, cv.COLOR_RGB2BGR  # pylint: disable=no-member
            )
            cv.imshow(window_name, display_image)  # pylint: disable=no-member
            cv.waitKey(1)  # pylint: disable=no-member

    def _get_current_state(self) -> ObjectCentricState:
        """Get the current object-centric observation."""
        assert self._current_state is not None, "Need to call reset() first"
        return self._current_state.copy()

    def _get_all_obs(self) -> dict[str, Any]:
        """Get the current raw observation (for compatibility with reward functions)."""
        assert self._robot_env is not None, "Robot environment not initialized"
        raw_obs = self._robot_env.get_obs()
        vec_obs = self._vectorize_observation(raw_obs)
        object_centric_state = self._get_object_centric_state()
        return {
            "vec": vec_obs,
            "object_centric_state": object_centric_state,
            "raw_obs": raw_obs,
        }

    def _get_object_centric_state(self) -> ObjectCentricState:
        """Get the current object-centric state of the environment."""
        # Collect object-centric data for all objects
        state_dict = {}
        for obj in self._objects:
            obj_state = obj.get_object_centric_state()
            state_dict.update(obj_state)
        for fixture in self._fixtures_dict.values():
            fixture_state = fixture.get_object_centric_state()
            state_dict.update(fixture_state)
        # Add robot into object-centric state.
        robot_state_dict = self._get_object_centric_robot_data()
        state_dict.update(robot_state_dict)
        return create_state_from_dict(state_dict, MujocoObjectTypeFeatures)

    def step_with_images(
        self, action: Array
    ) -> tuple[ObjectCentricState, float, bool, bool, dict[str, Any], dict[str, Any]]:
        """Step the environment and return object-centric observation."""
        # Run the action through the underlying environment
        assert self._robot_env is not None, "Robot environment not initialized"
        self._robot_env.step(action)

        # Update object-centric state
        self._current_state = self._get_object_centric_state()

        # Get raw observation for reward calculation
        all_obs = self._get_all_obs()

        # Visualization loop for rendered image
        if self.show_images:
            camera_images = self._robot_env.get_camera_images()
            if camera_images is not None:
                for camera_name in self._robot_env.camera_names:
                    if camera_name in camera_images:
                        self._visualize_image_in_window(
                            camera_images[camera_name],
                            f"TidyBot {camera_name} camera",
                        )

        # Calculate reward and termination
        reward = self.reward(all_obs)
        terminated = self._is_terminated(all_obs)
        truncated = False

        return self._get_current_state(), reward, terminated, truncated, {}, all_obs

    def step(
        self, action: Array
    ) -> tuple[ObjectCentricState, float, bool, bool, dict[str, Any]]:
        """Step the environment and return object-centric observation."""
        # Run the action through the underlying environment
        assert self._robot_env is not None, "Robot environment not initialized"
        self._robot_env.step(action)

        # Update object-centric state
        self._current_state = self._get_object_centric_state()

        # Get raw observation for reward calculation
        all_obs = self._get_all_obs()

        # Visualization loop for rendered image
        if self.show_images:
            camera_images = self._robot_env.get_camera_images()
            if camera_images is not None:
                for camera_name in self._robot_env.camera_names:
                    if camera_name in camera_images:
                        self._visualize_image_in_window(
                            camera_images[camera_name],
                            f"TidyBot {camera_name} camera",
                        )

        # Calculate reward and termination
        reward = self.reward(all_obs)
        terminated = self._is_terminated(all_obs)
        truncated = False

        return self._get_current_state(), reward, terminated, truncated, {}

    def _check_goals(self) -> bool:
        """Check if the goal has been achieved."""
        state = self._get_current_state()

        # Get all goal predicates, and determine if they should
        # be combined with "and" or "or"
        goal_predicates = self.task_config.get("goal_state", [])

        if len(goal_predicates) == 0:
            return False

        if goal_predicates[0] == "or":
            goal_conjunction = "or"
            goal_predicates = goal_predicates[1:]
        elif goal_predicates[0] == "and":
            goal_conjunction = "and"
            goal_predicates = goal_predicates[1:]
        else:
            goal_conjunction = "and"

        # Evaluate each goal predicate
        successes = []
        for pred in goal_predicates:
            if pred[0] in ["on", "in"]:
                # Treating "on" and "in" the same in predicate checking for now
                if pred[0] == "in":
                    print(
                        "Warning: Found 'in' predicate for success check, "
                        "which is not supported. Falling back to 'on' predicate."
                    )

                obj_name = pred[1]
                region_name = pred[2]
                obj = state.get_object_from_name(obj_name)

                # Handle robot objects specially (they have pos_base_x/y instead
                # of x/y/z)
                if obj_name == self.robot_name:
                    position = np.array(
                        [
                            state.get(obj, "pos_base_x"),
                            state.get(obj, "pos_base_y"),
                            0.0,  # Robot base is on the ground
                        ],
                        dtype=np.float32,
                    )
                else:
                    position = np.array(
                        [
                            state.get(obj, "x"),
                            state.get(obj, "y"),
                            state.get(obj, "z"),
                        ],
                        dtype=np.float32,
                    )
                region_config = self.task_config["regions"][region_name]

                if region_config["target"] == "ground":
                    # Check pose directly on the ground in the world frame
                    assert (
                        self._ground_fixture is not None
                    ), "Ground fixture not initialized"
                    in_region = self._ground_fixture.check_in_region(
                        position, region_name, self._robot_env
                    )
                else:
                    # Check first in fixtures, then in objects
                    target = region_config["target"]
                    entity: MujocoFixture | MujocoObject
                    if target in self._fixtures_dict:
                        entity = self._fixtures_dict[target]
                        in_region = entity.check_in_region(
                            position, region_name, self._robot_env
                        )
                    elif target in self._objects_dict:
                        entity = self._objects_dict[target]
                        in_region = entity.check_in_region(
                            position, region_name, self._robot_env
                        )
                    else:
                        raise ValueError(
                            f"Target '{target}' not found in fixtures or objects"
                        )

                successes.append(in_region)
            elif pred[0] == "balanced":
                # Check if a seesaw object is balanced (beam is horizontal)
                # Format: ["balanced", "seesaw_name", tolerance_degrees]
                obj_name = pred[1]
                tolerance_degrees = float(pred[2]) if len(pred) > 2 else 5.0

                # Get the seesaw object
                if obj_name not in self._objects_dict:
                    raise ValueError(f"Object '{obj_name}' not found for balance check")

                seesaw_obj = self._objects_dict[obj_name]

                if not isinstance(seesaw_obj, GeneratedSeesaw):
                    raise ValueError(
                        f"Object '{obj_name}' is not a GeneratedSeesaw, "
                        f"got {type(seesaw_obj).__name__}"
                    )

                is_balanced = seesaw_obj.is_balanced(tolerance_degrees)
                successes.append(is_balanced)
            else:
                raise NotImplementedError(
                    f"Goal predicate {pred[0]} not implemented in _check_goals"
                )

        if goal_conjunction == "and":
            return all(successes)
        if goal_conjunction == "or":
            return any(successes)
        raise ValueError(f"Unknown goal conjunction: {goal_conjunction}")

    def reward(self, obs: dict[str, Any]) -> float:
        """Calculate reward based on task completion."""
        return self._reward_calculator.calculate_reward(obs)

    def _is_terminated(self, obs: dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        # pylint: disable=unused-argument
        return self._check_goals()

    def render_all_cameras(self) -> dict[str, NDArray[np.uint8]]:
        """Render all available cameras in the environment."""
        if self.render_mode == "rgb_array":
            assert self._robot_env is not None, "Robot environment not initialized"
            images = self._robot_env.get_camera_images()
            if images is not None:
                return images
            raise RuntimeError("No camera images available in observation.")
        raise NotImplementedError(f"Render mode {self.render_mode} not supported")

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the environment."""
        if self.render_mode == "rgb_array":
            assert self._robot_env is not None, "Robot environment not initialized"
            images = self._robot_env.get_camera_images()
            if images is not None:
                image_keys = [k.split("_image")[0] for k in images.keys()]
                if self._render_camera_name and self._render_camera_name in image_keys:
                    return images[f"{self._render_camera_name}_image"]
                # Otherwise, return the first available image.
                for _, value in images.items():
                    return value
            raise RuntimeError("No camera image available in observation.")
        raise NotImplementedError(f"Render mode {self.render_mode} not supported")

    def close(self) -> None:
        """Close the environment."""
        if self.show_images:
            # Close OpenCV windows
            cv.destroyAllWindows()  # pylint: disable=no-member
        if self._robot_env is not None:
            self._robot_env.close()

    def set_render_camera(self, camera_name: str | None) -> None:
        """Set the camera to use for rendering."""
        self._render_camera_name = camera_name

    @abc.abstractmethod
    def _get_object_centric_robot_data(self) -> dict[Object, dict[str, float]]:
        """Get object-centric data for the robot.

        This method should be implemented by subclasses to provide robot-specific state
        data.
        """

    @abc.abstractmethod
    def _set_robot_state(self, state: ObjectCentricState) -> None:
        """Set the robot state in the simulation.

        This method should be implemented by subclasses to set the robot's state in the
        simulation.
        """


class ObjectCentricTidyBot3DEnv(ObjectCentricRobotEnv):
    """TidyBot-specific implementation of object-centric robot environment."""

    def _create_action_space(  # type: ignore
        self, config: TidyBot3DConfig
    ) -> Space[Array]:
        """Create action space for TidyBot's control interface."""
        return TidyBot3DRobotActionSpace()

    def _get_object_centric_robot_data(self) -> dict[Object, dict[str, float]]:
        assert self.robot_type == "tidybot"
        assert self._robot_env is not None, "Robot environment not initialized"
        robot = Object(self.robot_name, MujocoTidyBotRobotObjectType)
        # Build this super explicitly, even though verbose, to be careful.
        assert self._robot_env.qpos is not None
        assert self._robot_env.qvel is not None
        state_dict = {}
        state_dict[robot] = {
            "pos_base_x": self._robot_env.qpos["base"][0],
            "pos_base_y": self._robot_env.qpos["base"][1],
            "pos_base_rot": self._robot_env.qpos["base"][2],
            "pos_arm_joint1": self._robot_env.qpos["arm"][0],
            "pos_arm_joint2": self._robot_env.qpos["arm"][1],
            "pos_arm_joint3": self._robot_env.qpos["arm"][2],
            "pos_arm_joint4": self._robot_env.qpos["arm"][3],
            "pos_arm_joint5": self._robot_env.qpos["arm"][4],
            "pos_arm_joint6": self._robot_env.qpos["arm"][5],
            "pos_arm_joint7": self._robot_env.qpos["arm"][6],
            "pos_gripper": self._robot_env.ctrl["gripper"][0] / 255.0,
            "vel_base_x": self._robot_env.qvel["base"][0],
            "vel_base_y": self._robot_env.qvel["base"][1],
            "vel_base_rot": self._robot_env.qvel["base"][2],
            "vel_arm_joint1": self._robot_env.qvel["arm"][0],
            "vel_arm_joint2": self._robot_env.qvel["arm"][1],
            "vel_arm_joint3": self._robot_env.qvel["arm"][2],
            "vel_arm_joint4": self._robot_env.qvel["arm"][3],
            "vel_arm_joint5": self._robot_env.qvel["arm"][4],
            "vel_arm_joint6": self._robot_env.qvel["arm"][5],
            "vel_arm_joint7": self._robot_env.qvel["arm"][6],
            "vel_gripper": self._robot_env.qvel["gripper"][0],
        }
        return state_dict

    def _set_robot_state(self, state: ObjectCentricState) -> None:
        """Set the robot state in the simulation."""
        assert self._robot_env is not None, "Robot environment not initialized"

        # Get robot by type instead of by name for flexibility
        robots = state.get_objects(MujocoTidyBotRobotObjectType)
        assert len(robots) == 1, f"Expected exactly 1 robot, got {len(robots)}"
        robot_obj = list(robots)[0]

        # Reset the robot base position.
        robot_base_pos = [
            state.get(robot_obj, "pos_base_x"),
            state.get(robot_obj, "pos_base_y"),
            state.get(robot_obj, "pos_base_rot"),
        ]
        assert self._robot_env.qpos is not None
        self._robot_env.qpos["base"][:] = robot_base_pos

        # Reset the robot arm position.
        robot_arm_pos = [state.get(robot_obj, f"pos_arm_joint{i}") for i in range(1, 8)]
        assert self._robot_env.qpos is not None
        self._robot_env.qpos["arm"][:] = robot_arm_pos

        # Reset the robot gripper position.
        gripper_pos = state.get(robot_obj, "pos_gripper")
        assert self._robot_env.ctrl is not None
        self._robot_env.ctrl["gripper"][:] = gripper_pos * 255.0

        # Reset the robot base velocity.
        robot_base_vel = [
            state.get(robot_obj, "vel_base_x"),
            state.get(robot_obj, "vel_base_y"),
            state.get(robot_obj, "vel_base_rot"),
        ]
        assert self._robot_env.qvel is not None
        self._robot_env.qvel["base"][:] = robot_base_vel

        # Reset the robot arm velocity.
        robot_arm_vel = [state.get(robot_obj, f"vel_arm_joint{i}") for i in range(1, 8)]
        assert self._robot_env.qvel is not None
        self._robot_env.qvel["arm"][:] = robot_arm_vel

        # Reset the robot gripper velocity.
        gripper_vel = state.get(robot_obj, "vel_gripper")
        assert self._robot_env.qvel is not None
        self._robot_env.qvel["gripper"][:] = gripper_vel


class TidyBot3DEnv(ConstantObjectKinDEREnv):
    """TidyBot env with a constant number of objects."""

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricTidyBot3DEnv:
        return ObjectCentricTidyBot3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return [o.name for o in sorted(exemplar_state)]

    def _create_env_markdown_description(self) -> str:
        """Create environment description (policy-agnostic)."""
        scene_description = ""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricTidyBot3DEnv)
        if env.scene_type == "ground":
            scene_description = (
                " In the 'ground' scene, objects are placed randomly on a flat "
                "ground plane."
            )

        return f"""A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {env.scene_type} with {env.num_objects} objects.{scene_description}

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)
"""

    def _create_in_context_examples(self) -> str:
        """Create in-context learning examples for the environment."""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricTidyBot3DEnv)

        # Determine task type from task_config_path
        # Extract task name from path: e.g., "SweepIntoDrawer3D" from
        # ".../SweepIntoDrawer3D/SweepIntoDrawer3D-o5.json"
        task_path = Path(env.task_config_path)
        task_name = (
            task_path.parent.name
            if task_path.parent.name != "tasks"
            else task_path.stem.rsplit("-", 1)[0]
        )

        # Check task type based on task name
        if "Shelf3D" in task_name:
            return self._create_shelf3d_examples()

        if "SweepIntoDrawer3D" in task_name or "Sweep" in task_name:
            return self._create_sweep3d_examples()

        # Default generic example for other tasks
        return """Example task specification:
- Initial state:
  - on block1 region1
  - on block2 region2
- Goal state:
  - on block1 region3
  - on block2 region4
In this example, the robot needs to move block1 from region1 to region3, and block2 from region2 to region4.
"""

    def _create_shelf3d_examples(self) -> str:
        """Create in-context examples specifically for Shelf3D tasks.

        Returns formatted examples with actual parameterized skills and reasoning.
        """
        # pylint: disable=line-too-long
        return """
**Example 1: Single Object Pick and Place**

Initial State:
- Robot: base position (0.0, 0.0), orientation 0° (facing forward), arm retracted, gripper open
- Cube1: position (0.6, 0.0, 0.05) on ground, red color, size 0.02m
- Cupboard_1: position (1.5, 0.0, 0.0), shelf fixture with 3 shelves

Goal: Place cube1 on shelf 2 of cupboard_1

Task Specification:
- Initial state: on(cube1, ground_1_object_init_region)
- Goal state: on(cube1, cupboard_1_cube_1_goal_region)

Strategy: Pick up cube from ground → Navigate to cupboard → Place cube on target shelf

High-Level Plan:
1. pick_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object)[0.5, 0.0]
   - Navigate robot base to approach cube1 from distance 0.5m
   - Approach angle 0.0 radians means approaching from directly in front
   - Lower arm to grasp height near ground level (~0.05m above ground)
   - Close gripper to grasp cube1
   - Retract arm to safe carrying configuration
   
   Detailed Steps:
   a) Base navigation: Move from (0.0, 0.0) to approximately (0.1, 0.0) 
      - Target distance parameter (0.5) positions robot 0.5m from cube
      - Target rotation parameter (0.0) aligns robot to face cube head-on
   b) Arm motion planning: Plan trajectory from retracted config to pre-grasp pose
      - End effector target: (0.39m forward, 0.0m lateral, -0.35m down from base)
      - Orientation: gripper faces down (quaternion: 0.707, 0.707, 0, 0)
   c) Gripper closing: Activate gripper closure (gripper_pos → 0.0)
   d) Arm retraction: Return to home configuration to lift cube safely

2. place_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.9, 0.0, -1.571]
   - Navigate to cupboard while holding cube1
   - Position relative to cupboard for shelf placement
   - Extend arm to place cube on target shelf
   - Open gripper to release cube
   - Retract arm to home position
   
   Detailed Steps:
   a) Base navigation: Move from pick location to cupboard approach position
      - Target distance parameter (0.9) positions robot 0.9m from cupboard center
      - Y-offset parameter (0.0) centers robot laterally with cupboard
      - Rotation parameter (-1.571 ≈ -π/2) orients robot perpendicular to cupboard
      - Robot moves while carrying cube1 (collision checking disabled for held object)
   b) Arm motion planning: Plan trajectory to shelf placement position
      - End effector target: (0.7m forward, 0.0m lateral, 0.0m height)
      - Orientation: gripper horizontal (quaternion: 0.5, 0.5, 0.5, 0.5)
      - Position aligns with target shelf height and depth
   c) Gripper opening: Release gripper (gripper_pos → 1.0)
   d) Arm retraction: Return to home configuration

Goal Reached:
- Cube1 successfully placed on cupboard_1 shelf 2 ✓
- Object within goal region tolerance (±0.3m x, ±0.12m y, ±0.03m z) ✓

Key Insights:
- pick_ground parameters [distance, rotation] control robot approach to ground objects
  - Distance typically 0.5-0.7m provides optimal reach without collision
  - Rotation in range [-π, π] determines approach angle relative to object
- place_ground parameters [distance, y_offset, rotation] control placement relative to fixture
  - Distance ~0.9m for cupboard positioning balances reach and stability
  - Y-offset allows lateral placement adjustment along fixture
  - Rotation -π/2 is standard for approaching cupboard from side
- Robot must navigate collision-free while carrying objects
- Arm trajectories planned in task space, IK solved in PyBullet simulation
- Gripper states: 0.0 = fully closed, 1.0 = fully open

**Example 2: Multiple Objects Sequential Placement**

Initial State:
- Robot: base position (0.0, 0.0), orientation 0°, arm retracted, gripper open
- Cube1: position (0.6, -0.1, 0.05) on ground
- Cube2: position (0.65, 0.1, 0.05) on ground
- Cupboard_1: position (1.5, 0.0, 0.0)

Goal: Place both cubes on cupboard shelves

Task Specification:
- Initial state: on(cube1, ground_region_1), on(cube2, ground_region_2)
- Goal state: on(cube1, cupboard_shelf_left), on(cube2, cupboard_shelf_right)

Strategy: Pick and place each cube sequentially

High-Level Plan:
1. pick_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object)[0.6, 0.0]
   - Approach cube1 from 0.6m distance at 0° angle
   - Lower arm and grasp cube1
   - Lift to carrying configuration

2. place_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.85, -0.1, -1.571]
   - Navigate to cupboard with cube1
   - Y-offset -0.1m places cube on left side of shelf
   - Place and release cube1

3. pick_ground(robot:mujoco_tidybot_robot, cube2:mujoco_movable_object)[0.6, 0.0]
   - Return to ground area and approach cube2
   - Approach from 0.6m distance at 0° angle
   - Grasp and lift cube2

4. place_ground(robot:mujoco_tidybot_robot, cube2:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.92, 0.1, -1.571]
   - Navigate to cupboard with cube2
   - Y-offset +0.1m places cube on right side of shelf
   - Place and release cube2

Goal Reached:
- Both cubes successfully placed on cupboard shelves ✓
- Objects spatially separated to avoid collision ✓

Key Insights:
- Sequential pick-and-place for multiple objects
- Y-offset parameter crucial for placing multiple objects without collision
- Distance parameter may vary (0.85m vs 0.92m) to optimize base positioning
- Base navigation automatically avoids objects on ground during returns

**Example 3: Alternative Approach Angles**

Initial State:
- Robot: base position (-0.1, -0.1), orientation 45°
- Cube1: position (0.5, 0.2, 0.05) on ground (reachable from multiple angles)
- Cupboard_1: position (1.5, 0.0, 0.0)

Goal: Place cube1 on cupboard

Strategy: Use non-zero approach angle for pick

High-Level Plan:
1. pick_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object)[0.5, 1.571]
   - Approach cube1 from distance 0.5m
   - Rotation parameter 1.571 (≈ π/2) approaches from side (90° angle)
   - This avoids obstacles that might be in direct approach path
   - Grasp mechanics identical regardless of approach direction

2. place_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.9, 0.0, -1.571]
   - Standard cupboard placement approach
   - Distance 0.9m, centered (y_offset=0.0), perpendicular orientation

Goal Reached:
- Cube1 placed successfully using alternative approach ✓
- Demonstrates navigation flexibility ✓

Key Insights:
- pick_ground rotation parameter enables approach from any angle
- Useful for obstacle avoidance or constrained spaces
- π/2 = side approach, π = rear approach, 0 = front approach
- Robot automatically computes collision-free base motion plans
- End effector grasp pose remains consistent regardless of base approach angle

**Strategy Selection Guidelines:**
- For objects on ground: Use pick_ground with distance ≈ 0.5m for optimal reach
- For cupboard placement: Use distance ≈ 0.9m and rotation ≈ -π/2
- For multiple objects: Adjust y_offset to prevent placement collisions (±0.1m separation)
- For obstacle avoidance: Vary rotation parameter in pick_ground to approach from clear path
- Navigation automatically handles collision avoidance with world boundaries and static fixtures
"""

    def _create_sweep3d_examples(self) -> str:
        """Create in-context examples specifically for SweepIntoDrawer3D tasks.

        Returns formatted examples with actual parameterized skills for sweeping tasks.
        """
        # pylint: disable=line-too-long
        return """
**Example 1: Open Drawer, Pick Wiper, Sweep Objects**

Initial State:
- Robot: base position (1.45, -0.25), orientation 180° (facing kitchen island), arm retracted, gripper open
- Kitchen_island_drawer: closed, located at shelf 1 of kitchen island at (0.5, 0.0, 0.5)
- Wiper_0: position (0.35, 0.0, 1.0) on kitchen_island shelf 2, T-shaped tool with handle height 0.20m, head length 0.20m
- Cube_0: position (0.15, -0.05, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_1: position (0.25, 0.03, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_2: position (0.30, -0.02, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_3: position (0.20, 0.08, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_4: position (0.40, -0.06, 1.0) on kitchen_island shelf 2, size 0.01m, red color

Goal: Sweep all five cubes from shelf 2 into the opened drawer at shelf 1

Task Specification:
- Initial state:
  - in(wiper_0, wiper_init_region)
  - in(cube_0, blocks_init_region), in(cube_1, blocks_init_region), in(cube_2, blocks_init_region)
  - in(cube_3, blocks_init_region), in(cube_4, blocks_init_region)
  - on(robot, robot_kitchen_init_region)
- Goal state:
  - on(cube_0, kitchen_island_shelf_1_partition_1_region)
  - on(cube_1, kitchen_island_shelf_1_partition_1_region)
  - on(cube_2, kitchen_island_shelf_1_partition_1_region)
  - on(cube_3, kitchen_island_shelf_1_partition_1_region)
  - on(cube_4, kitchen_island_shelf_1_partition_1_region)

Strategy: Open drawer → Pick up wiper tool → Sweep all cubes into opened drawer

High-Level Plan:
1. open_drawer(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.7, -3.142]
   - Navigate robot to kitchen island drawer
   - Grasp the drawer handle
   - Pull drawer open to allow object placement
   - Release handle and retract arm
   
   Detailed Steps:
   a) Base navigation: Move from (1.45, -0.25) to drawer approach position
      - Target distance parameter (0.7) positions robot 0.7m from drawer center
      - Target rotation parameter (-3.142 ≈ -π) orients robot to face drawer directly
      - Navigation avoids collision with kitchen island and other fixtures
   b) Arm motion planning: Plan trajectory to drawer handle grasp pose
      - End effector target: (0.07m forward, 0.3m lateral, -0.12m down from handle)
      - Orientation: gripper angled for handle (roll: -π - π/16, pitch: 0, yaw: -π/2)
      - Transform: DRAWER_TRANSFORM_TO_OBJECT for initial grasp
   c) Gripper closing: Activate gripper closure (gripper_pos → 0.0) to grasp handle
   d) Drawer opening motion: Pull drawer outward along slide rail
      - End effector moves from initial grasp (0.07m) to extended position (0.18m)
      - Transform: DRAWER_TRANSFORM_TO_OBJECT_END for final position
      - Smooth trajectory following to maintain handle grasp
      - Drawer opens approximately 0.11m to provide access
   e) Gripper opening: Release handle (gripper_pos → 1.0)
   f) Arm retraction: Return to home configuration safely
   
   Action Steps: 300 steps total
   - Navigation: ~100 steps to approach drawer
   - Grasp and open: ~150 steps for arm motion, grasping, pulling
   - Retraction: ~50 steps to return to home

2. pick_wiper(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.7, -3.142]
   - Navigate to wiper location on kitchen_island shelf 2
   - Approach wiper with specific orientation for tool grasping
   - Grasp wiper handle
   - Lift wiper to carrying configuration
   
   Detailed Steps:
   a) Base navigation: Move from drawer location to wiper position
      - Target distance parameter (0.7) positions robot 0.7m from wiper
      - Target rotation parameter (-3.142 ≈ -π) aligns robot perpendicular to shelf
      - Base must position for reaching elevated shelf 2 (~1.0m height)
   b) Arm motion planning: Plan trajectory to wiper handle grasp
      - End effector target: Wiper handle at (0.35, 0.0, 1.0) world coordinates
      - Grasp transform: WIPER_TRANSFORM_TO_OBJECT offset (0.02m, 0, 0.03m)
      - Orientation: angled grip for T-shaped handle (roll: -π - π/16, pitch: 0, yaw: -π/2)
      - IK solution computed for 7-DOF arm to reach elevated position
   c) Gripper closing: Close gripper (gripper_pos → 0.0) around handle
   d) Arm retraction: Lift wiper and return to stable carrying configuration
      - Collision checking accounts for wiper geometry (handle + head)
      - Held object transform: base_link_to_held_obj computed for wiper
   
   Action Steps: 300 steps total
   - Navigation: ~100 steps to approach shelf 2
   - Grasp: ~120 steps for arm motion and grasping
   - Lift: ~80 steps to retract with tool

3. sweep(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.55, -3.142]
   - Position robot for sweep motion across shelf 2
   - Execute sweeping motion with wiper to push all cubes
   - Cubes fall from shelf 2 into opened drawer at shelf 1 below
   
   Detailed Steps:
   a) Base navigation: Move to sweep starting position
      - Target distance parameter (0.55) positions robot 0.55m from sweep target
      - Target rotation parameter (-3.142 ≈ -π) maintains perpendicular orientation
      - Position allows full range of motion for sweep trajectory
   b) Sweeping motion execution: Multi-waypoint end-effector path
      - Waypoint 1: WIPER_SWEEP_TRANSFORM - (-0.05m, -0.1m, 0.025m) start position
        * Wiper positioned at edge of shelf 2, slightly left of cubes
        * End effector at ~1.0m height matching shelf surface
      - Waypoint 2: WIPER_SWEEP_TRANSFORM_END - (0.15m, 0.05m, 0.025m) mid-sweep
        * Wiper sweeps rightward and forward across shelf
        * Pushes cubes toward drawer opening
        * Smooth trajectory maintains contact with shelf surface
      - Waypoint 3: WIPER_SWEEP_TRANSFORM_END_2 - (0.28m, 0.15m, 0.025m) final position
        * Completes sweep motion
        * Ensures all cubes pushed off shelf into drawer below
   c) Follow end-effector path: Smooth trajectory execution
      - Use smoothly_follow_end_effector_path() for Cartesian motion
      - IK solved at each step to track waypoints
      - Held wiper collision geometry checked throughout motion
   d) Arm retraction: Return to home after sweep completion
   
   Action Steps: 200 steps total
   - Navigation: ~50 steps to sweep start position
   - Sweep motion: ~130 steps for multi-waypoint trajectory
   - Retraction: ~20 steps to stabilize

Goal Reached:
- Drawer successfully opened to receive objects ✓
- Wiper tool acquired for manipulation ✓
- All five cubes swept from shelf 2 into shelf 1 drawer ✓
- Objects within goal region tolerance inside drawer partition ✓

Key Insights:
- open_drawer parameters [distance, rotation] control approach to drawer fixture
  - Distance typically 0.6-0.8m for reaching drawer handle
  - Rotation ≈ -π positions robot facing drawer directly
  - Handle grasp uses specialized DRAWER_TRANSFORM_TO_OBJECT offset
  - Pulling motion extends along drawer slide rail direction
- pick_wiper parameters [distance, rotation] control approach to elevated shelf object
  - Distance 0.6-0.8m balances reach and stability for shelf 2 (~1.0m height)
  - Rotation ≈ -π maintains perpendicular approach to shelf
  - Wiper grasp uses WIPER_TRANSFORM_TO_OBJECT for T-handle geometry
  - Carrying requires collision checking for extended tool geometry
- sweep parameters [distance, rotation] control sweep execution positioning
  - Distance 0.5-0.7m provides optimal range for sweeping motion
  - Rotation ≈ -π maintains alignment with shelf edge
  - Multi-waypoint trajectory: start → mid-sweep → end position
  - Waypoints: WIPER_SWEEP_TRANSFORM → TRANSFORM_END → TRANSFORM_END_2
  - Smooth Cartesian path following ensures contact with shelf surface
- Object parameters include all entities for collision avoidance:
  - robot, wiper, drawer, and all cubes (cube_0 through cube_4)
- Action spaces: 11-dimensional continuous control
  - Base: [vx, vy, omega] for mobile navigation
  - Arm: 7-DOF joint velocities for manipulation
  - Gripper: 1-DOF for open/close
- Physics simulation: PyBullet used for motion planning and IK
  - Collision-free trajectory planning with RRT-based planner
  - Inverse kinematics for end-effector target poses
  - Dynamic simulation validates skill execution

**Example 2: Alternative Sweep Parameters**

Initial State:
- Similar configuration to Example 1
- Cubes distributed differently on shelf 2: more clustered on right side
- Drawer already open (pre-opened scenario)

Goal: Pick wiper and sweep cubes into drawer (drawer opening not needed)

Strategy: Skip drawer opening → Pick wiper → Sweep with adjusted parameters

High-Level Plan:
1. pick_wiper(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.65, -3.142]
   - Approach wiper from 0.65m distance
   - Standard perpendicular orientation (-π rotation)
   - Grasp and lift wiper tool

2. sweep(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.60, -3.0]
   - Approach from 0.60m distance (slightly closer)
   - Rotation parameter -3.0 (≈ -π + 0.14) provides slight angle adjustment
   - Angled approach better aligns with clustered cube distribution
   - Same multi-waypoint sweep trajectory
   - All cubes successfully pushed into drawer

Goal Reached:
- Wiper acquired ✓
- Alternative sweep parameters adapt to object distribution ✓
- All cubes in drawer ✓

Key Insights:
- Drawer opening can be skipped if drawer already accessible
- Sweep distance parameter adjustable: 0.5-0.7m range
- Sweep rotation parameter allows angle tuning: -π ± 0.2 radians
- Parameter adjustment adapts to different object clustering patterns
- Same waypoint trajectory achieves sweep regardless of approach angle

**Example 3: Detailed Parameter Space Understanding**

Parameter Ranges:
- open_drawer: [distance, rotation]
  - distance: typically 0.6-0.8m (optimal range for reaching drawer handle)
  - rotation: -π to π (standard -π for direct frontal approach)
- pick_wiper: [distance, rotation]
  - distance: typically 0.6-0.8m (balances reach for elevated shelf)
  - rotation: -π to π (standard -π for perpendicular shelf approach)
- sweep: [distance, rotation]
  - distance: typically 0.5-0.7m (provides sweep motion range)
  - rotation: -π to π (standard -π, slight variations for angle adjustment)

Action Space Details:
- TidyBot3D robot: 11-dimensional continuous action space
  - Base motion: [vx, vy, omega] ∈ [-0.1, 0.1]³ m/s and rad/s
  - Arm joints: 7-DOF velocities within joint limits (rad/s)
  - Gripper: 1-DOF position control ∈ [0.0, 1.0] (0=closed, 1=open)

Physical Constraints:
- Kitchen island fixture: fixed position, provides shelf 1 and shelf 2 surfaces
- Drawer: slide rail mechanism, opens ~0.11m outward
- Wiper: T-shaped tool, handle 0.20m × 0.015m, head 0.20m × 0.03m
- Cubes: small 0.01m blocks, low mass, easily pushed by wiper head
- World bounds: navigation constrained by room boundaries and fixtures
- Collision avoidance: active for all objects and fixtures during motion planning

Execution Timing:
- open_drawer: ~300 action steps (30 seconds at 10Hz control)
- pick_wiper: ~300 action steps (30 seconds)
- sweep: ~200 action steps (20 seconds)
- Total task: ~800 action steps (80 seconds) for Example 1

**Strategy Selection Guidelines:**
- For drawer opening: Use distance ≈ 0.7m, rotation ≈ -π for frontal approach
- For wiper picking: Use distance ≈ 0.7m, rotation ≈ -π for shelf 2 access
- For sweeping: Use distance ≈ 0.55-0.60m, rotation ≈ -π with slight adjustment for object distribution
- Multi-step tasks: Execute skills sequentially with state updates between skills
- Tool manipulation: Account for held object geometry in collision checking
- Trajectory planning: Use PyBullet IK and motion planning for collision-free paths
- Parameter sampling: Use sample_parameters() with different seeds for variation
- Object parameters: Include all entities (robot, wiper, drawer, all cubes) for complete scene awareness
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has variants that differ in scene type and number of objects. Scene types include 'ground', 'cabinet', etc. The number of objects varies across variants."

    def _create_variant_specific_description(self) -> str:
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricTidyBot3DEnv)
        scene_type = env.scene_type
        num_objects = env.num_objects
        obj_str = "1 object" if num_objects == 1 else f"{num_objects} objects"
        return f"This variant uses the '{scene_type}' scene type with {obj_str}."

    def _create_obs_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observation includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: positions and orientations of all objects
- Camera images: RGB images from base and wrist cameras
- Scene-specific features: handle positions for cabinets/drawers
"""

    def _create_action_markdown_description(self) -> str:
        """Create action space description."""
        return """Actions control:
- base_pose: [x, y, theta] - Mobile base position and orientation
- arm_pos: [x, y, z] - End effector position in world coordinates
- arm_quat: [x, y, z, w] - End effector orientation as quaternion
- gripper_pos: [pos] - Gripper open/close position (0=closed, 1=open)
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRobotEnv)
        if env.scene_type == "ground":
            return (
                "The primary reward is for successfully placing objects at their "
                "target locations.\n"
                "- A reward of +1.0 is given for each object placed within a 5cm "
                "tolerance of its target.\n"
                "- A smaller positive reward is given for objects within a 10cm "
                "tolerance to guide the robot.\n"
                "- A small negative reward (-0.01) is applied at each timestep to "
                "encourage efficiency.\n"
                "The episode terminates when all objects are placed at their "
                "respective targets.\n"
            )
        return """Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
"""


class ObjectCentricRBY1A3DEnv(ObjectCentricRobotEnv):
    """RBY1A-specific implementation of object-centric robot environment."""

    def _create_action_space(  # type: ignore
        self, config: TidyBot3DConfig
    ) -> Space[Array]:
        """Create action space for TidyBot's control interface."""
        return RBY1ARobotActionSpace()

    def _get_object_centric_robot_data(self) -> dict[Object, dict[str, float]]:
        assert self.robot_type == "rby1a"
        assert self._robot_env is not None, "Robot environment not initialized"
        robot = Object(self.robot_name, MujocoRBY1ARobotObjectType)
        # Build this super explicitly, even though verbose, to be careful.
        state_dict = {}
        assert self._robot_env.qpos is not None
        state_dict[robot] = {
            "pos_base_right": self._robot_env.qpos["base"][0],
            "pos_base_left": self._robot_env.qpos["base"][1],
            # TODO add more attributes  # pylint: disable=fixme
        }
        return state_dict

    def _set_robot_state(self, state: ObjectCentricState) -> None:
        """Set the robot state in the simulation."""
        assert self._robot_env is not None, "Robot environment not initialized"

        # Get robot by type instead of by name for flexibility
        robots = state.get_objects(MujocoRBY1ARobotObjectType)
        assert len(robots) == 1, f"Expected exactly 1 robot, got {len(robots)}"
        robot_obj = list(robots)[0]

        # Reset the robot base position.
        assert self._robot_env.qpos is not None
        robot_base_pos = [
            state.get(robot_obj, "pos_base_right"),
            state.get(robot_obj, "pos_base_left"),
        ]
        self._robot_env.qpos["base"][:] = robot_base_pos

        # TODO add more attributes  # pylint: disable=fixme


class RBY1A3DEnv(ConstantObjectKinDEREnv):
    """RBY1A env with a constant number of objects."""

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricRBY1A3DEnv:
        return ObjectCentricRBY1A3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return [o.name for o in sorted(exemplar_state)]

    def _create_env_markdown_description(self) -> str:
        """Create environment description (policy-agnostic)."""
        scene_description = ""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRBY1A3DEnv)
        if env.scene_type == "ground":
            scene_description = (
                " In the 'ground' scene, objects are placed randomly on a flat "
                "ground plane."
            )

        return f"""A 3D mobile manipulation environment using the RBY1A platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {env.scene_type} with {env.num_objects} objects.{scene_description}

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has variants that differ in scene type and number of objects. Scene types include 'ground', 'cabinet', etc. The number of objects varies across variants."

    def _create_variant_specific_description(self) -> str:
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRBY1A3DEnv)
        scene_type = env.scene_type
        num_objects = env.num_objects
        obj_str = "1 object" if num_objects == 1 else f"{num_objects} objects"
        return f"This variant uses the '{scene_type}' scene type with {obj_str}."

    def _create_obs_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observation includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: positions and orientations of all objects
- Camera images: RGB images from base and wrist cameras
- Scene-specific features: handle positions for cabinets/drawers
"""

    def _create_action_markdown_description(self) -> str:
        """Create action space description."""
        return """Actions control:
- base_pose: [x, y, theta] - Mobile base position and orientation
- arm_pos: [x, y, z] - End effector position in world coordinates
- arm_quat: [x, y, z, w] - End effector orientation as quaternion
- gripper_pos: [pos] - Gripper open/close position (0=closed, 1=open)
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRobotEnv)
        if env.scene_type == "ground":
            return (
                "The primary reward is for successfully placing objects at their "
                "target locations.\n"
                "- A reward of +1.0 is given for each object placed within a 5cm "
                "tolerance of its target.\n"
                "- A smaller positive reward is given for objects within a 10cm "
                "tolerance to guide the robot.\n"
                "- A small negative reward (-0.01) is applied at each timestep to "
                "encourage efficiency.\n"
                "The episode terminates when all objects are placed at their "
                "respective targets.\n"
            )
        return """Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """TODO
"""

    def _create_in_context_examples(self) -> str:
        """Create in-context examples for the environment."""
        return """**Example: Simple Manipulation Task**

Initial State:
- Robot: RBY1A with closed gripper

Goal: Execute task-specific manipulation.

Steps: Identify targets, plan paths, grasp, transport, place.
"""

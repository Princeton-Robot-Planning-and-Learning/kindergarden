"""Collect human demonstrations in kinder environments."""

import argparse
import math
import sys
import time
from pathlib import Path

import dill as pkl
import numpy as np

try:
    import pygame
except ImportError:
    print("Error: pygame is required for demo collection.")
    print("Install it with: pip install pygame")
    sys.exit(1)

import kinder
from docs.generate_env_docs import sanitize_env_id


class AnalogStick:
    """Represents a single analog stick for GUI input."""

    def __init__(self, center_x: int, center_y: int, radius: int = 30) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.x = 0.0
        self.y = 0.0
        self.is_active = False

    def update_from_mouse(
        self, mouse_pos: tuple[int, int], mouse_pressed: bool
    ) -> None:
        """Update stick position based on mouse input."""
        if not mouse_pressed:
            self.is_active = False
            self.x = 0.0
            self.y = 0.0
            return

        if self.is_mouse_over(mouse_pos):
            self.is_active = True
            mouse_x, mouse_y = mouse_pos
            dx = mouse_x - self.center_x
            dy = mouse_y - self.center_y
            self.x = dx / self.radius
            self.y = -dy / self.radius
        else:
            self.is_active = False

    def is_mouse_over(self, mouse_pos: tuple[int, int]) -> bool:
        """Check if mouse is over this analog stick."""
        mouse_x, mouse_y = mouse_pos
        dx = mouse_x - self.center_x
        dy = mouse_y - self.center_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance <= self.radius

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the analog stick on the screen."""
        pygame.draw.circle(
            screen, (100, 100, 100), (self.center_x, self.center_y), self.radius, 2
        )
        pygame.draw.line(
            screen,
            (150, 150, 150),
            (self.center_x - 5, self.center_y),
            (self.center_x + 5, self.center_y),
            1,
        )
        pygame.draw.line(
            screen,
            (150, 150, 150),
            (self.center_x, self.center_y - 5),
            (self.center_x, self.center_y + 5),
            1,
        )
        if self.is_active:
            stick_x = self.center_x + int(self.x * self.radius)
            stick_y = self.center_y - int(self.y * self.radius)
            pygame.draw.circle(screen, (255, 255, 255), (stick_x, stick_y), 8)
        else:
            pygame.draw.circle(
                screen, (200, 200, 200), (self.center_x, self.center_y), 8
            )


class DemoCollector:
    """Collect human demonstrations in kinder environments using pygame."""

    def __init__(
        self,
        env_id: str,
        demo_dir: Path,
        screen_width: int = 1000,
        screen_height: int = 600,
        render_fps: int = 20,
        font_size: int = 24,
        start_seed: int = 0,
    ) -> None:
        self.env_id = env_id
        self.demo_dir = demo_dir
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        kinder.register_all_environments()
        self.env = kinder.make(env_id, render_mode="rgb_array")
        self.unwrapped_env = self.env.unwrapped
        if not hasattr(self.unwrapped_env, "get_action_from_gui_input"):
            raise RuntimeError(
                f"Environment {env_id} must implement get_action_from_gui_input."
            )
        self.observations: list = []
        self.actions: list = []
        self.rewards: list[float] = []
        self.terminated: bool = False
        self.truncated: bool = False
        self.keys_pressed: set[str] = set()
        pygame.init()
        pygame.joystick.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Demo Collection - {env_id}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, font_size)
        self.render_fps = render_fps
        self.controller = None
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"✓ Controller connected: {self.controller.get_name()}")
            print(f"  Axes: {self.controller.get_numaxes()}")
            print(f"  Buttons: {self.controller.get_numbuttons()}")
        else:
            print("⚠ No controller detected - using mouse/keyboard controls")
        stick_radius = 40
        side_panel_width = 150
        self.left_stick = AnalogStick(
            side_panel_width // 2, screen_height // 2, stick_radius
        )
        self.right_stick = AnalogStick(
            screen_width - side_panel_width // 2, screen_height // 2, stick_radius
        )
        self.side_panel_width = side_panel_width
        self.seed = start_seed
        self.reset_env()

    def reset_env(self) -> None:
        """Reset the environment and start collecting a new demo."""
        obs, _ = self.env.reset(seed=self.seed)
        self.seed += 1
        self.observations = [obs]
        self.actions = []
        self.rewards = []
        self.terminated = False
        self.truncated = False
        self.keys_pressed.clear()

    def save_demo(self) -> None:
        """Save the current demo to disk."""
        if not self.observations or not self.actions:
            print("Warning: No demo data to save!")
            return
        timestamp = int(time.time())
        demo_seed = self.seed - 1
        demo_subdir = self.demo_dir / sanitize_env_id(self.env_id) / str(demo_seed)
        demo_subdir.mkdir(parents=True, exist_ok=True)
        demo_path = demo_subdir / f"{timestamp}.p"
        demo_data = {
            "env_id": self.env_id,
            "timestamp": timestamp,
            "seed": demo_seed,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        with open(demo_path, "wb") as f:
            pkl.dump(demo_data, f)
        print(f"Demo saved to {demo_path}")

    def handle_events(self) -> bool:
        """Handle pygame events."""
        if self.terminated or self.truncated:
            if self.terminated:
                print("✓ Goal reached! Auto-saving demo and resetting...")
                self.save_demo()
            else:
                print("⚠ Episode truncated. Not saving demo. Resetting...")
            self.reset_env()
        assert not (self.terminated or self.truncated)
        some_action_input = False
        if self.controller:
            left_x = self.controller.get_axis(0)
            left_y = -self.controller.get_axis(1)
            right_x = self.controller.get_axis(2)
            right_y = -self.controller.get_axis(3)
            deadzone = 0.1
            if abs(left_x) < deadzone:
                left_x = 0.0
            if abs(left_y) < deadzone:
                left_y = 0.0
            if abs(right_x) < deadzone:
                right_x = 0.0
            if abs(right_y) < deadzone:
                right_y = 0.0
            self.left_stick.x = left_x
            self.left_stick.y = left_y
            self.left_stick.is_active = abs(left_x) > 0 or abs(left_y) > 0
            self.right_stick.x = right_x
            self.right_stick.y = right_y
            self.right_stick.is_active = abs(right_x) > 0 or abs(right_y) > 0
            if self.left_stick.is_active or self.right_stick.is_active:
                some_action_input = True
        else:
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            left_stick_clicked = (
                self.left_stick.is_mouse_over(mouse_pos) and mouse_pressed
            )
            right_stick_clicked = (
                self.right_stick.is_mouse_over(mouse_pos) and mouse_pressed
            )
            self.left_stick.update_from_mouse(mouse_pos, left_stick_clicked)
            self.right_stick.update_from_mouse(mouse_pos, right_stick_clicked)
            if left_stick_clicked or right_stick_clicked:
                some_action_input = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_env()
                elif event.key == pygame.K_g:
                    self.save_demo()
                elif event.key == pygame.K_q:
                    return False
                else:
                    key_name = pygame.key.name(event.key)
                    if key_name not in {"r", "g", "q"}:
                        self.keys_pressed.add(key_name)
                        some_action_input = True
            if event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                if key_name in self.keys_pressed:
                    self.keys_pressed.discard(key_name)
                    some_action_input = True
            if event.type == pygame.JOYBUTTONDOWN and self.controller:
                if event.button == 0:
                    self.keys_pressed.add("space")
                    some_action_input = True
                elif event.button == 1:
                    self.reset_env()
                elif event.button == 3:
                    self.save_demo()
                elif event.button == 11:
                    self.keys_pressed.add("w")
                    some_action_input = True
                elif event.button == 12:
                    self.keys_pressed.add("s")
                    some_action_input = True
            if event.type == pygame.JOYBUTTONUP and self.controller:
                if event.button == 0:
                    self.keys_pressed.discard("space")
                    some_action_input = True
                elif event.button == 11:
                    self.keys_pressed.discard("w")
                    some_action_input = True
                elif event.button == 12:
                    self.keys_pressed.discard("s")
                    some_action_input = True
        if some_action_input:
            self.step_env()
        return True

    def render(self) -> None:
        """Render the image in the GUI."""
        img: np.ndarray = self.env.render()  # type: ignore
        assert len(img.shape) == 3 and img.shape[2] in (3, 4)  # type: ignore
        assert img.dtype == np.uint8  # type: ignore
        img_surface = pygame.surfarray.make_surface(img[:, :, :3].swapaxes(0, 1))
        img_rect = img_surface.get_rect()
        center_width = self.screen_width - 2 * self.side_panel_width
        scale = min(center_width / img_rect.width, self.screen_height / img_rect.height)
        new_width = int(img_rect.width * scale)
        new_height = int(img_rect.height * scale)
        img_surface = pygame.transform.scale(img_surface, (new_width, new_height))
        img_rect = img_surface.get_rect()
        img_rect.center = (self.screen_width // 2, self.screen_height // 2)
        self.screen.fill((0, 0, 0))
        left_panel = pygame.Rect(0, 0, self.side_panel_width, self.screen_height)
        right_panel = pygame.Rect(
            self.screen_width - self.side_panel_width,
            0,
            self.side_panel_width,
            self.screen_height,
        )
        pygame.draw.rect(self.screen, (0, 0, 0), left_panel)
        pygame.draw.rect(self.screen, (0, 0, 0), right_panel)
        self.screen.blit(img_surface, img_rect)
        self.left_stick.draw(self.screen)
        self.right_stick.draw(self.screen)
        left_label = self.font.render("Left Stick", True, (255, 255, 255))
        right_label = self.font.render("Right Stick", True, (255, 255, 255))
        self.screen.blit(
            left_label, (self.left_stick.center_x - 40, self.left_stick.center_y - 60)
        )
        self.screen.blit(
            right_label,
            (self.right_stick.center_x - 40, self.right_stick.center_y - 60),
        )
        status_text = f"{self.env_id} - Demo Length: {len(self.actions)}"
        text_surface = self.font.render(status_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.topleft = (10, 10)
        self.screen.blit(text_surface, text_rect)
        if self.controller:
            instructions = [
                "Left Stick: Rotate robot",
                "Right Stick: Move robot (x,y)",
                "X Button: Toggle vacuum",
                "D-pad Up/Down: Extend/retract arm",
                "Circle: Reset | Square: Save",
            ]
        else:
            instructions = [
                "Mouse: Click analog sticks",
                "W/S: Extend/retract arm",
                "Space: Toggle vacuum",
                "R: Reset | G: Save | Q: Quit",
            ]
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, (200, 200, 200))
            text_rect = text_surface.get_rect()
            text_rect.bottomleft = (
                10,
                self.screen_height - 10 - (len(instructions) - i - 1) * 25,
            )
            self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def step_env(self) -> None:
        """Step the environment one time."""
        left_x, left_y = self.left_stick.x, self.left_stick.y
        right_x, right_y = self.right_stick.x, self.right_stick.y
        input_data = {
            "keys": self.keys_pressed,
            "left_stick": (left_x, left_y),
            "right_stick": (right_x, right_y),
        }
        assert hasattr(self.unwrapped_env, "get_action_from_gui_input")
        action = self.unwrapped_env.get_action_from_gui_input(input_data)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.terminated = terminated
        self.truncated = truncated

    def run(self) -> None:
        """Run the demo collection GUI."""
        running = True
        while running:
            running = self.handle_events()
            self.render()
            self.clock.tick(self.render_fps)

        pygame.quit()


def _main() -> None:
    """Parse command line arguments and run demo collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("demos"),
    )
    args = parser.parse_args()
    if not args.env_id.startswith("kinder/"):
        print("Error: Environment ID must start with 'kinder/'")
        return
    collector = DemoCollector(args.env_id, args.demo_dir)
    collector.run()


if __name__ == "__main__":
    _main()

"""A CGL context for headless OpenGL rendering on macOS."""

from mujoco.cgl import GLContext


class CGLGLContext(GLContext):
    """A CGL context for headless OpenGL rendering on macOS.

    Unlike the GLFW context, this creates no window. macOS only allows NSWindow on the
    main thread, so a GLFW context aborts the process outright when a render is driven
    from a worker thread -- which is how anything serving an env renders.
    """

    def __init__(
        self, max_width, max_height, device_id=-1
    ):  # pylint: disable=unused-argument
        super().__init__(max_width, max_height)

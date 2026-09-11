# Local Tossing room preview

This prototype extends the lab2 room front to x=4.2 and extends its side walls
to meet it. The rear wall and chamfered corners keep the original layout.
The six walls use one task-owned layout, `tossing_room.xml`, in both background
modes. Fancy mode replaces the corresponding decorative walls and reuses their
material; plain mode shows white walls. The layout's `floor_visual_size` hint
extends the rendered floor without changing the infinite collision plane.

The original barrier remains 10 m wide, 20 cm high, and 6 cm thick, near x=1.3.
It is now a static `FixedCuboid` fixture. Its original small XY spawn range is
preserved, with an explicit center height of 0.1 m. Wall and barrier contact
settings are task-local; robot physics and action limits are unchanged.

Both counts retain the movable receiver, its x=[2.6, 3.42], y=[-2.3, 2.3]
spawn range, cube spawns, and ordinary bin-relative goals. The nearest front-wall
face at x=4.19 leaves at least 0.62 m beyond the receiver's maximum outer x=3.57.

This is a local visual and physics prototype, not a completed baseline update
or a proof that every reset requires and permits a successful toss. Existing
planners need to recognize the barrier's fixture type and include the explicit
room walls in their collision scenes. The room walls are static scene geometry,
not additional object-centric observations; the shared XML is their source of
geometry. The receiver remains a movable object. No controller retuning is
included here.

Render the registered `kinder/Tossing3D-o1-v0` or `-o2-v0` normally with
`scene_bg=True` (fancy) or `scene_bg=False` (plain). The task camera provides an
elevated overview. Intended throw validation and broader bypass checks remain
separate from this preview.

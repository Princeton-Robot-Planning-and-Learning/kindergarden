# FragileTossing3D

`kinder/FragileTossing3D-o1-v0` (also o2) is an opt-in Tossing3D variant.
The original Tossing3D registrations, success predicate, controls, and physics are unchanged.

A floor-aligned square follows each bin's x/y position and yaw. It is rendered in
teal with no collision masks or added mass. It may overlap the wall and robot.
`mat_size` defaults to 4 metres; `damage_cost` defaults to 10 per bare-ground
contact onset. Both are configurable and validated. Walls and bins are not ground.

Every physics step during robot execution checks actual cube/ground-plane contacts.
Contact points within a protective square (including its edge) are free. Multiple
points on the same cube count once. Persistent bare-ground contact is charged only
on entry; separation and a new contact can incur another charge. Entering bare
ground while sliding off the mat also starts a bare-ground contact episode.
Initial placement and human reset settling are exempt until the cube is lifted
clear of the floor. These exemptions are not a damage detector: damage always
requires a physics contact. General object-state restoration also establishes a
new placement exemption, so it should not be used as uncharged robot execution.

`step(...)[4]` contains `damage_events`, the step's summed `damage_cost`, and
`fragile_object` configuration. Each event includes cube name, contact position,
simulation time, a per-reset event sequence, and cost. Rewards remain task rewards;
consumers must explicitly add damage to their cost objective. No physical breakage
is simulated. A mat overlapping the robot side makes that ground legitimately free.

Tests cover real drops, resting contact, renewed impacts, placement exemptions,
short lifts, transformed mat boundaries, and parameter validation.

# Fauna Sprout model (not distributed)

The Sprout environments load the robot from this folder. The model is not public, so
git ignores all files here except this README. `MANIFEST.in` also keeps this folder
out of the source distribution.

Put the vendor model (v0.1.4) into this folder. Do not edit the files:

```
fauna_sprout/
    sprout.xml
    meshes/        (collision/ and visual/)
    textures/
```

Without `sprout.xml`, the Sprout environments are not registered.

The robot parameters that the vendor model does not have (motors, gains, default pose,
command delay) are public. They are in `robots/sprout_config.json`.

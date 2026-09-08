# ScoopPour3D-o20

## Usage

```python
import kinder

kinder.register_all_environments()
env = kinder.make("kinder/ScoopPour3D-o20-v0")
```

## Description

Transfer 20 cubes from the source bin to the green receiving bin on the kitchen
island. A scoop is available. This variant uses the same robot, bins, cube sizes,
placement ranges and physical parameters as the other ScoopPour3D counts.

The action representation and object-centric observation features follow the
[ScoopPour3D family](../../ScoopPour3D.md), with 20 cube objects in the observation.

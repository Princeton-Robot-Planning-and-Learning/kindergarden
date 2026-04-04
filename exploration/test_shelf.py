import kinder
kinder.register_all_environments()
import gymnasium
env = gymnasium.make("kinder/KinematicShelf3D-o2-v0")
obs, info = env.reset(seed=123)
print(obs.shape)  # Should print something like (50,)
env.close()
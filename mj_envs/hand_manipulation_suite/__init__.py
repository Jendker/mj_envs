from gym.envs.registration import register
from mj_envs.mujoco_env import MujocoEnv

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

# Relocate an object to the target (Shadow Robot)
register(
    id='relocateSR-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvSRV0',
    max_episode_steps=10000,
)
from mj_envs.hand_manipulation_suite.relocateSR_v0 import RelocateEnvSRV0

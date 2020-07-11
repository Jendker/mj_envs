from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

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
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relocate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

# Remove an object from the initial position
register(
    id='remove-v0',
    entry_point='mj_envs.hand_manipulation_suite:RemoveEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.remove_v0 import RemoveEnvV0

# Remove an object from the initial position with old reward function - toy example
register(
    id='remove_old_reward-v0',
    entry_point='mj_envs.hand_manipulation_suite:RemoveOldRewardEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.remove_old_reward_v0 import RemoveOldRewardEnvV0

# Custom environment - relocate an object to the target
register(
    id='relocate_custom-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateCustomEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_custom_v0 import RelocateCustomEnvV0

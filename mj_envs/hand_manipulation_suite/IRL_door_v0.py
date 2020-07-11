from envs.irl_env import IRL_env
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0


class IRLDoorEnvV0(DoorEnvV0, IRL_env):
    def __init__(self, **kwargs):
        self.irl_model = None
        DoorEnvV0.__init__(self)
        IRL_env.__init__(self, **kwargs)

    def step(self, a):
        original_action = a.copy()
        ob, original_reward, done, infos = DoorEnvV0.step(self, a)
        reward = self.get_reward(ob, original_action)
        infos['original_reward'] = original_reward

        return ob, reward, done, infos

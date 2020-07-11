from envs.irl_env import IRL_env
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0


class IRLHammerEnvV0(HammerEnvV0, IRL_env):
    def __init__(self, **kwargs):
        self.irl_model = None
        HammerEnvV0.__init__(self)
        IRL_env.__init__(self, **kwargs)

    def step(self, a):
        original_action = a.copy()
        ob, _, done, infos = HammerEnvV0.step(self, a)
        reward = self.get_reward(ob, original_action)

        return ob, reward, done, infos

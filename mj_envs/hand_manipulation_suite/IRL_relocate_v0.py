from envs.irl_env import IRL_env
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0


class IRLRelocateEnvV0(RelocateEnvV0, IRL_env):
    def __init__(self, **kwargs):
        self.irl_model = None
        RelocateEnvV0.__init__(self)
        IRL_env.__init__(self, **kwargs)

    def step(self, a):
        original_action = a.copy()
        ob, _, done, infos = RelocateEnvV0.step(self, a)
        reward = self.get_reward(ob, original_action)

        return ob, reward, done, infos

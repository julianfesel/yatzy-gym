from gym.envs.registration import register

register(
    id='Yatzy-v0',
    entry_point='gym_yatzy.envs:YatzyEnv',
)
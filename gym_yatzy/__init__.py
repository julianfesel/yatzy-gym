from gymnasium.envs.registration import register

register(
    id='yatzy-v0',
    entry_point='gym_yatzy.envs:YatzyEnv',
)

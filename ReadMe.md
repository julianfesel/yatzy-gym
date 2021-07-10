# Yatzy Gym

An openai gym environment to train agents in a version of poker dice. There are a lot of different names and variants of
this game. You can check these Wikipedia entries for additional information:

- [Yatzy](https://en.wikipedia.org/wiki/Yatzy)
- [Yahtzee](https://en.wikipedia.org/wiki/Yahtzee)
- [Kniffel (german)](https://de.wikipedia.org/wiki/Kniffel)
- [Poker Dice](https://en.wikipedia.org/wiki/Poker_dice)

## The rules of this environment

The rules that this environment implements are the same as explained [here](https://en.wikipedia.org/wiki/Yatzy).

## The environment

### Observation space

The observation space is defined by this python code:

```python
from gym import spaces

observation_space = spaces.Dict({
    'dice_values': spaces.MultiDiscrete([6] * 5),  # The eye values of the current dice
    'number_rolls': spaces.Discrete(3),  # How many times the dice were rolled this turn
    'set_fields': spaces.MultiBinary(16),  # Which fields are already set
    'sum_eye_fields': spaces.Box(0, 106, shape=(1,), dtype=int),  # Total reward of the eye fields (excl. bonus)
    'total_field_value': spaces.Box(0, 500, shape=(1,), dtype=int)  # Total reward of all fields (incl. bonus)
})
```

### Action space

By default the action space is given by this python code:

```python
from gym import spaces

action_space = spaces.Dict({
    'dice_mask': spaces.Tuple([spaces.Discrete(2)] * 5),
    'fill_field': spaces.Discrete(2),
    'field_to_fill': spaces.Discrete(15)
})
```

### Optional configuration

There are two parameters that can be given to the environment via a config dict for optional customization:

- 'wrong_move_punishment' (default -10): Changes the "reward" for an action against the rules of the game. For example
  if the agent tries to roll the dice a fourth time
- 'discrete_action' (default False): Changes the action space to be a discrete space with 46 levels. The first 31
  correspond to a specific dice mask for rolling the dice and the remaining 15 to the filling of one of the boxes. 
  
## References
- [Custom OpenAI Gym environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
- [Publishing a package](https://realpython.com/pypi-publish-python-package/)
- [Gym Environment for Yahtzee (there is a difference to Yatze)](https://github.com/villebro/gym-yahtzee)
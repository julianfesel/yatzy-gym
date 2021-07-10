import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import sys
from typing import List, Dict, Tuple, Union, Any, Optional


class YatzyEnv(gym.Env):
    """
    A gym environment implementing the poker dice game Yatzy.
    """
    metadata = {'render.modes': ['human']}

    # Define observation space
    observation_space = spaces.Dict({
        'dice_values': spaces.MultiDiscrete([6] * 5),  # The eye values of the current dice
        'number_rolls': spaces.Discrete(3),  # How many times the dice were rolled this turn
        'set_fields': spaces.MultiBinary(16),  # Which fields are already set
        'sum_eye_fields': spaces.Box(0, 106, shape=(1,), dtype=int),  # Total reward of the eye fields (excl. bonus)
        'total_field_value': spaces.Box(0, 500, shape=(1,), dtype=int)  # Total reward of all fields (incl. bonus)
    })

    def __init__(self, env_config: Optional[Dict] = None):
        """
        Initialisation of the environment.

        :param env_config: A dictionary that can be used to change some aspects of the environment
        'discrete_action': Whether to use a discrete action space (default uses a dictionary)
        'wrong_move_punishment': The size of the punishment (in reward) for actions of the agent that
        violate the rules (default -10)
        """
        if env_config is None:
            self.env_config = {}
        else:
            self.env_config = env_config

        # Set whether to use a discrete action space (default uses a dictionary)
        self.discrete_actions = self.env_config.get('discrete_action', False)

        # Set punishment for actions of the agent that violate the rules:
        self.wrong_move_punishment = self.env_config.get('wrong_move_punishment', -10)

        # Set reward range
        self.reward_range = (self.wrong_move_punishment, 71)

        # Define action space depending on fiven parameter
        if self.discrete_actions:
            self.action_space = spaces.Tuple([spaces.Discrete(31 + 15)])
        else:
            self.action_space = spaces.Dict({
                'dice_mask': spaces.Tuple([spaces.Discrete(2)] * 5),
                'fill_field': spaces.Discrete(2),
                'field_to_fill': spaces.Discrete(15)
            })

        # Dictionary containing dice masks corresponding to certain actions -
        self.action_dice_masks = dict(
            [(num - 1, np.array(list(f'{num:05b}'), dtype=int).astype(bool)) for num in range(1, 32)])

        # Define template for human readable render output
        self.render_template = [
            "Current Dice:        ",
            "| {} | {} | {} | {} | {} |",
            "Current Fields:      "
        ]

        # Define a list of names for the fields
        self.field_names = ['{}s'.format(ind) for ind in range(1, 7)]
        self.field_names += [
            '63bonus',
            'pair',
            'two pair',
            'three equal',
            'four equal',
            'house',
            'small straight',
            'large straight',
            'chance',
            'yatze'
        ]

        # Create a list of field names with right format for rendering
        self.field_names_format = [item + '{0:.>' + str(21 - len(item)) + '}' for item in self.field_names]

        # Initialise other variables with placeholders
        self.env = {}
        self.last_action = {}
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None) -> List[int]:
        """
        Sets the seed for random number generation.
        :param seed: Optionally fix seed for debugging.
        :return: List of seeds with first one being the main seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: spaces.dict) -> Tuple[Dict, Union[int, Any], Union[np.array, bool], Dict]:
        """
        The step functin of the environment.
        :param action: An action provided by the agent
        :return: Tuple consisting of:
            observation: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls will return undefined results
            info: an empty dictionary in this environment
        """
        # Check whether action takes on the right form
        assert self.action_space.contains(action)

        # Extract relevant information from the action into a dictionary
        dice_mask, fill_field, field_to_fill = self.normalize_action(action)
        self.last_action = {
            'dice_mask': dice_mask,
            'fill_field': fill_field,
            'field_to_fill': field_to_fill
        }

        # Either fill a field or roll the selected dice, depending on chosen action
        if fill_field:  # take action to fill one of the fields
            reward = self.fill_eye_fields(field_to_fill)
            self.reset_turn()
        else:  # take action to roll some of the dice again
            reward = self.roll_dice(dice_mask)
        # Make a new observation after taking the step
        obs = self.make_obs()

        # Check whether game has ended
        done = np.all(self.env['field_rewards'] > -1)  # Check if all fields are filled

        return obs, reward, done, {}

    def make_obs(self) -> Dict:
        """
        A method to create an observation from the current environment
        :return: A dictionary containing the observation
        """
        # Get bool arrey specifying fields that are already set
        set_fields = self.env.get('field_rewards') > -1

        # Same as before only for dice eye fields
        set_dice_fields = self.env.get('field_rewards')[0:6] > -1

        # Create the observation dictionary
        obs = {
            'dice_values': self.env.get('dice') - 1,
            'number_rolls': self.env.get('number_rolls') - 1,
            'set_fields': set_fields.astype(int),
            'sum_eye_fields': np.array([
                self.env.get('field_rewards')[0:6][set_dice_fields].sum()
            ]),
            'total_field_value': np.array([
                self.env.get('field_rewards')[set_fields].sum()
            ])
        }

        # Check that new observation has the right form
        assert self.observation_space.contains(obs)

        return obs

    def normalize_action(self, action: spaces.tuple) -> Tuple[np.array, bool, int]:
        """
        Resolves the action space coming from the agent into the internal format of the step function.
        :param action: the action from the agent

        :return:
        a tuple containing:
        a mask describing which dice should be rolled;
        a boolean determining whether the agent wants to fill a field;
        an integer determining the field to fill
        """
        if self.discrete_actions:
            chosen_action = action[0]
            if chosen_action < 31:  # Case were agent wants to roll the dice
                dice_mask = self.action_dice_masks[chosen_action]
                fill_field = False
                field_to_fill = -1
            else:  # Case were agent wants to fill a field
                fill_field = True
                field_to_fill = chosen_action - 31
                dice_mask = None
        else:
            dice_mask = np.array(action['dice_mask'])
            fill_field = action['fill_field']
            field_to_fill = action['field_to_fill']
        # take care of bonus field
        if field_to_fill > 5:
            field_to_fill += 1
        return dice_mask, fill_field, field_to_fill

    def fill_eye_fields(self, field_to_fill: int) -> Union[float, int]:
        """
        Function to handle process of filling a field. Changes the environment and returns the appropriate reward
        :param field_to_fill: Integer determining the field to fill from action of agent
        :return: the reward for filling the field
        """
        # Rename temporarily for code readability
        field_rewards = self.env['field_rewards']
        dice = self.env['dice']

        # If field is already filled return punishment, else fill field
        if field_rewards[field_to_fill] > -1:
            reward = self.wrong_move_punishment
        else:
            # handle all six number of eye fields
            if field_to_fill < 6:
                selected_eye = field_to_fill + 1
                reward = (dice == selected_eye).sum() * selected_eye
                field_rewards[field_to_fill] = reward
                if field_rewards[6]:
                    if field_rewards[0:6].sum() > 62:
                        reward += 50
                        field_rewards[6] = 50
                    elif np.all(field_rewards[0:5] > -1):
                        field_rewards[6] = 0
                    else:
                        pass
            # handle special fields
            elif field_to_fill == 7:  # pair
                eye_counts = np.bincount(dice)
                if np.any(eye_counts >= 2):
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 8:  # two pairs
                eye_counts = np.bincount(dice)
                if (eye_counts >= 2).sum() == 2: # The pairs must be different
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 9:  # three of the same
                eye_counts = np.bincount(dice)
                eyes_on_dice = eye_counts.argmax()
                if eye_counts[eyes_on_dice] >= 3:
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 10:  # four of the same
                eye_counts = np.bincount(dice)
                eyes_on_dice = eye_counts.argmax()
                if eye_counts[eyes_on_dice] >= 4:
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 11:  # full house
                eye_counts = np.bincount(dice)
                if (np.any(eye_counts == 2)) and (np.any(eye_counts == 3)):
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 12:  # small straight
                if np.all(dice == np.arange(1, 6)):
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 13:  # large straight
                if np.all(dice == np.arange(2, 7)):
                    reward = dice.sum()
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            elif field_to_fill == 14:  # chance
                reward = dice.sum()
                field_rewards[field_to_fill] = reward
            elif field_to_fill == 15:  # yatzy
                eye_counts = np.bincount(dice)
                if np.any(eye_counts == 5):
                    reward = 50
                    field_rewards[field_to_fill] = reward
                else:
                    reward = 0
                    field_rewards[field_to_fill] = reward
            else:
                reward = self.wrong_move_punishment

        return reward

    def roll_dice(self, dice_mask: np.ndarray) -> Union[int, float]:
        """
        Method to implement the dice rolling action. Takes in boolean mask to select dice to be rolled. Creates new
        random integers for these dice and then returns the dice eye values sorted from low to high.
        :param dice_mask: boolean dice mask
        :return: The reward for the action
        """
        dice_new = self.env["dice"]

        # Punish if trying to roll dice after three rolls
        if self.env["number_rolls"] == 3:
            return self.wrong_move_punishment
        else:
            num_selected_dice = dice_mask.sum()

            # Punish if no dice are selected for rolling
            if num_selected_dice == 0:
                return self.wrong_move_punishment
            # Dice rolling logic
            else:
                new_eyes = np.random.randint(1, 7, num_selected_dice)
                dice_new[dice_mask.astype(bool)] = new_eyes
                dice_new.sort()
                self.env["number_rolls"] += 1
                reward = 0

        # Set new dice values in environment
        self.env["dice"] = dice_new
        return reward

    def reset_turn(self) -> None:
        """
        A method to reset the environment after filling a field
        :return: None
        """
        self.env['dice'] = np.sort(np.random.randint(1, 7, 5))
        self.env['number_rolls'] = 1

    def reset(self) -> Dict:
        """
        Method to reset the environment after finishing an episode.
        :return: The first observation after resetting
        """

        # Reset env values
        self.env = {
            'dice': np.sort(np.random.randint(1, 7, 5)),
            'number_rolls': 1,
            'field_rewards': np.full(shape=16, fill_value=-1)
        }

        # Reset action with dummies
        self.last_action = {
            'dice_mask': np.ones(5),
            'fill_field': 0,
            'field_to_fill': None
        }

        # Create first observation
        obs = self.make_obs()
        return obs

    def render(self, mode: str = 'human') -> None:
        """
        Renders current state and previous action to stdout
        :param mode: currently only 'human' is supported
        :return: None - writes directly to stdout
        """
        # Check for right mode
        if mode != 'human':
            raise NotImplementedError('Currently only human mode is supported')
        out = self.render_template.copy()

        # Create output strings
        # Fill dice entries
        out[1] = out[1].format(*self.env['dice'])

        # Fill field rewards
        field_rewards = [string.format(reward) for string, reward in
                         zip(self.field_names_format, self.env['field_rewards'])]
        out += field_rewards

        # Generate string for action information depending on rolling or field filling
        action_info = [" "*21]*2

        if self.last_action['fill_field']:
            name_index = self.last_action['field_to_fill']
            action_info += [
                'Action:{0:.>14}'.format('FIELD'),
                '{0:.>21}'.format(self.field_names[name_index]),
            ]
        else:
            action_info += [
                "Taken Action:        ",
                'Action:{0:.>14}'.format('ROLL'),
                "| {} | {} | {} | {} | {} |".format(*self.last_action['dice_mask'].astype(int)),
            ]
        out = action_info + out

        # Write to stdout
        sys.stdout.write("\n".join(["".join(row) for row in out]) + "\n")

    def close(self):
        pass


if __name__ == '__main__':
    env = YatzyEnv()
    episode_done = False

    while not episode_done:
        current_action = env.action_space.sample()
        (current_obs, current_reward, episode_done, _) = env.step(current_action)
        env.render()

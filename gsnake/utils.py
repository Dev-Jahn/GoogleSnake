from collections import OrderedDict
from enum import IntEnum, unique, auto

import numpy as np
from gym.spaces import Box, Dict


@unique
class SnakeState(IntEnum):
    """
    Enum representing the state of a cell in the grid
    May add more states in the future
    Start from 1 for compatibility
    """
    # SNAKE direction is always
    SNAKE_U = 1
    SNAKE_R = auto()
    SNAKE_D = auto()
    SNAKE_L = auto()
    FOOD = auto()
    ANTI_FOOD = auto()
    OBSTACLE = auto()
    EMPTY = auto()

    def __str__(self):
        return f'{self.name}<{self.value}>'

    def __repr__(self):
        return f'{self.name}<{self.value}>'

    @classmethod
    def n_states_internal(cls):
        """
        Get the number of all internal states
        :return: The number of all internal states including all directions
        """
        return len(cls)

    @classmethod
    def n_states_external(cls):
        """
        Number of states that are visible to the agent
        :return: number of states including SNAKE, EMPTY, FOOD and etc
        """
        return len([enum for enum in SnakeState if not enum.name.startswith('SNAKE')]) + 1


@unique
class SnakeAction(IntEnum):
    """
    Enum representing the action of the snake
    """
    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    # Only for internal calculations
    BACKWARD = 3

    @staticmethod
    def absolute_direction(direction, action):
        """
        Get the direction of the snake after taking the given action
        :param direction: Direction of the snake before taking the action
        :param action: Action to take
        :return: Direction of the snake after taking the action
        """
        # return (direction + action) % 4
        return (direction + action - 2) % 4 + 1  # gui compat

    @staticmethod
    def absolute_position(direction, action):
        """
        Get the lambda function to calculate absolute position of the snake after taking an action
        :param direction: The direction of the snake
        :param action: The action to take
        :return: The lambda function takes row, col as input and relative position as output
        """
        # print('DIR:', direction)
        # print('ACT:', action)
        # print('ABS DIR:', SnakeAction.absolute_direction(direction, action))
        if SnakeAction.absolute_direction(direction, action) == SnakeState.SNAKE_U:
            return lambda row, col: (row - 1, col)
        elif SnakeAction.absolute_direction(direction, action) == SnakeState.SNAKE_R:
            return lambda row, col: (row, col + 1)
        elif SnakeAction.absolute_direction(direction, action) == SnakeState.SNAKE_D:
            return lambda row, col: (row + 1, col)
        elif SnakeAction.absolute_direction(direction, action) == SnakeState.SNAKE_L:
            return lambda row, col: (row, col - 1)
        else:
            raise ValueError


class OneHotBox(Box):
    """
    A box that can be one-hot encoded
    """

    def __init__(self, n, dtype=np.uint8):
        super().__init__(low=0, high=1, shape=(n,), dtype=dtype)

    def sample(self):
        return np.eye(self.shape[0], dtype=self.dtype)[np.random.randint(self.shape[0])]

    def contains(self, x):
        return np.array_equal(x, np.eye(self.shape[0], dtype=self.dtype)[np.argmax(x)])

    def __repr__(self):
        return "OneHotBox({})".format(self.shape)


class SnakeNode:
    """
    Node of the snake
    """

    def __init__(self, row, col, direction: SnakeState):
        self.pos = row, col
        self.direction = direction
        self.prev_node = None
        self.next_node = None

    @property
    def row(self):
        return self.pos[0]

    @property
    def col(self):
        return self.pos[1]

    def __add__(self, other):
        """
        Overloaded operator +
        Attaches two nodes together
        Calculate the direction of attached node based on direction of the current node and position of the other node
        :param other: The other node
        :return: The other node
        """
        assert isinstance(other, SnakeNode)
        assert self.row == other.row or self.col == other.col
        if self.row == other.row:
            if self.col < other.col:
                other.direction = SnakeState.SNAKE_L
            else:
                other.direction = SnakeState.SNAKE_R
        else:
            if self.row < other.row:
                other.direction = SnakeState.SNAKE_U
            else:
                other.direction = SnakeState.SNAKE_D
        other.prev_node = self
        self.next_node = other
        return other

    def __next__(self):
        return self.next_node

    def __repr__(self):
        return f"Snake<{self.row}, {self.col}, {str(self.direction)}>"

    def destroy(self):
        """
        Unlink self from prev_node and next_node
        Only used to destroy the tail node for now
        """
        if self.prev_node is not None:
            self.prev_node.next_node = None
        if self.next_node is not None:
            self.next_node.prev_node = None


class AppleNode:
    """
        Node of the apple
    """

    def __init__(self, direction, anti_direction, *pos):
        self.direction = direction
        self.anti_direction = anti_direction
        self.prev_node = None
        self.next_node = None
        self.pos = pos[0:2]
        self.anti_pos = pos[2:]

    def __add__(self, other):
        """
        Overloaded operator +
        Attaches two nodes together
        Calculate the direction of attached node based on direction of the current node and position of the other node
        :param other: The other node
        :return: The other node
        """
        other.prev_node = self
        self.next_node = other
        return other

    def __repr__(self):
        return f"Apple<{self.pos}, {self.anti_pos}, {str(self.direction)}>"


class SnakeGrid:
    """
    Grid class for the GoogleSnake game environment
    """

    def __init__(self, config, init_length=2, seed=None):
        self.config = config
        self.init_length = init_length
        self.seed = seed
        self.grid = np.zeros(config.grid_shape, dtype=np.uint8)

        self.poisoned = False
        self.poison_count = 0
        self.poison_count_max = 5

        # Cursor of the head and tail SnakeNode
        self.head = None
        self.tail = None

        # Cursor of the head and tail apple(food)
        self.first_food = None
        self.last_food = None

        self.anti_food_check = self.config.portal or self.config.poison
        self.next_portal_position = None

    def reset(self):
        """
        Reset the grid to all empty and spawn objects
        :return:
        """
        self.poisoned = False
        self.poison_count = 0

        self.head = None
        self.tail = None

        self.first_food = None
        self.last_food = None

        self.next_portal_position = None

        self.grid.fill(SnakeState.EMPTY)
        self._init_snake()
        self.generate_food(n=self.config.n_foods, new=True)

    def _init_snake(self):
        """
        Initialize the snake in the grid
        :return: None
        """
        start_pos = self.config.start_pos
        start_dir = self.config.start_dir

        # TODO: Add support for different initial snake positions
        self.head = SnakeNode(*start_pos, direction=start_dir)
        cursor = self.head
        for i in range(self.init_length - 1):
            cursor = cursor + SnakeNode(
                *SnakeAction.absolute_position(start_dir, SnakeAction.BACKWARD)(*start_pos),
                direction=start_dir
            )
        self.tail = cursor

        # Draw
        cursor = self.head
        while cursor is not None:
            self.grid[cursor.pos] = cursor.direction
            cursor = cursor.next_node

    def generate_food(self, n=1, new=False):
        """
        Generate food in an empty cell
        :return: None
        """
        coords = self._sample_empty(n=2 * n if self.config.portal or self.config.poison else n)
        if self.anti_food_check:    coords = tuple(x + y for x, y in zip(coords[0::2], coords[1::2]))

        if (self.last_food is not None) and (not new):
            self.grid[coords[0][0:2]] = SnakeState.FOOD
            if self.anti_food_check:    self.grid[coords[0][2:4]] = SnakeState.ANTI_FOOD
            cursor = self.last_food
            cursor = cursor + AppleNode(np.random.randint(1, 5), np.random.randint(1, 5), *coords[0])
        else:
            self.grid[coords[0][0:2]] = SnakeState.FOOD
            if self.anti_food_check:    self.grid[coords[0][2:4]] = SnakeState.ANTI_FOOD
            self.first_food = AppleNode(np.random.randint(1, 5), np.random.randint(1, 5), *coords[0])
            cursor = self.first_food

        for coord in coords[1:]:
            if self.anti_food_check:    self.grid[coord[2:4]] = SnakeState.ANTI_FOOD
            self.grid[coord[0:2]] = SnakeState.FOOD
            cursor = cursor + AppleNode(np.random.randint(1, 5), np.random.randint(1, 5), *coord)
        self.last_food = cursor

    def move_apple(self):
        cursor = self.first_food
        while cursor is not None:
            cursor.direction, cursor.pos = self.next_apple_position(cursor.direction, cursor.pos, anti=False)
            if self.config.portal or self.config.poison:
                cursor.anti_direction, cursor.anti_pos = self.next_apple_position(cursor.anti_direction,
                                                                                  cursor.anti_pos, anti=True)
            cursor = cursor.next_node

    def next_apple_position(self, direction, pos, anti=False):
        plus_row = [0, -1, 0, 1, 0]
        plus_col = [0, 0, 1, 0, -1]

        dir, reverse_dir = direction, (direction + 1) % 4 + 1
        new_pos = pos[0] + plus_row[dir], pos[1] + plus_col[dir]
        new_reverse_pos = pos[0] + plus_row[reverse_dir], pos[1] + plus_col[reverse_dir]
        if not self._off_grid(*new_pos) and self.grid[new_pos] == SnakeState.EMPTY:
            self.grid[pos] = SnakeState.EMPTY
            self.grid[new_pos] = SnakeState.ANTI_FOOD if anti else SnakeState.FOOD
            return dir, new_pos
        elif not self._off_grid(*new_reverse_pos) and self.grid[new_reverse_pos] == SnakeState.EMPTY:
            self.grid[pos] = SnakeState.EMPTY
            self.grid[new_reverse_pos] = SnakeState.ANTI_FOOD if anti else SnakeState.FOOD
            return reverse_dir, new_reverse_pos
        return dir, pos

    def generate_obstacles(self, n=1):
        """
        Generate obstacles in empty cells
        There should be no obstacles around the 8 directions.
        :return: None
        """
        if not self._check_obstacles_available:  return
        coords = self._sample_empty_obstacles(n=n)
        for coord in coords:
            self.grid[coord] = SnakeState.OBSTACLE

    def _sample_empty(self, n=1):
        """
        Sample a random empty cell from the grid
        :return: Tuples of ((row1, col1), (row2, col2), ...)
        """
        empty = np.argwhere(self.grid == SnakeState.EMPTY)
        return tuple(tuple(row) for row in empty[np.random.randint(len(empty), size=n)])

    def _sample_empty_obstacles(self, n=1):
        """
        Sample a random empty cell from the grid for obstacles
        :return: Tuples of ((row1, col1), (row2, col2), ...)
        """
        result = list()
        generate_num = 0
        H, W = self.config.grid_shape
        px = [-1, 0, 1, 1, 1, 0, -1, -1]
        py = [-1, -1, -1, 0, 1, 1, 1, 0]
        while generate_num < n:
            x, y, flag = np.random.randint(W), np.random.randint(H), 0
            if not self.grid[y, x] == SnakeState.EMPTY:   continue
            for i in range(8):
                nx, ny = x + px[i], y + py[i]
                if nx < 0 or nx >= W:   continue
                if ny < 0 or ny >= H:   continue
                if self.grid[ny, nx] == SnakeState.OBSTACLE:    flag = 1
                if (ny, nx) in result:  flag = 1
            if not flag:
                result.append((y, x))
                generate_num += 1
        return tuple(result)

    def _check_obstacles_available(self):
        """
        There is a limit to the number of obstacles,
        :return: Boolean
        """
        H, W = self.config.grid_shape
        obstacles_num = len(np.argwhere(self.grid == SnakeState.OBSTACLE))
        if H * W // (obstacles_num * 11) == 0:
            return True
        else:
            return False

    def closest_food_dist(self):
        """
        Find distance between the closest food to the head of the snake
        :return: Euclidean distance between the head and the closest food
        """
        food = np.argwhere(self.grid == SnakeState.FOOD)
        head = np.array(self.head.pos)
        return np.min(np.linalg.norm(food - head, axis=1))

    def reverse_snake(self):
        """
        if reverse mode,
        head and tail are flipped when snake eats food
        """
        cursor = self.tail
        dir, dir_before = self.grid[cursor.pos], self.grid[cursor.pos]
        while True:
            cursor.direction = (dir + 1) % 4 + 1
            self.grid[cursor.pos] = cursor.direction
            cursor.prev_node, cursor.next_node = cursor.next_node, cursor.prev_node
            cursor = cursor.next_node
            if cursor == None:  break
            dir, dir_before = dir_before, self.grid[cursor.pos]
        self.head, self.tail = self.tail, self.head

    def get_snakes(self):
        """
        Get the positions of all the snake cells
        :return: List of tuples of (row, col) of all snake cells
        """
        return np.argwhere(self.grid >= SnakeState.SNAKE_U).tolist()

    def get_foods(self):
        """
        Get the positions of all the food cells
        :return: List of tuples of (row, col) of all food cells
        """
        return np.argwhere(self.grid == SnakeState.FOOD).tolist()

    def get_anti_foods(self):
        """
        Get the positions of all the anti food cells
        :return: List of tuples of (row, col) of all anti food cells
        """
        return np.argwhere(self.grid == SnakeState.ANTI_FOOD).tolist()

    def get_obstacles(self):
        """
        Get the positions of all the obstacles
        :return: List of tuples of (row, col) of all food cells
        """
        return np.argwhere(self.grid == SnakeState.OBSTACLE).tolist()

    def is_snake(self, row, col):
        """
        Check if the given cell is part of the snake
        :param row: Row of the cell
        :param col: Column of the cell
        :return: True if the cell is part of the snake, False otherwise
        """
        return self.grid[row, col] in (SnakeState.SNAKE_U, SnakeState.SNAKE_R, SnakeState.SNAKE_D, SnakeState.SNAKE_L)

    def is_head(self, row, col):
        """
        Check if the given cell is the head of the snake
        :param row: Row of the head
        :param col: Column of the head
        :return: True if the cell is the head, False otherwise
        """
        return self.head.pos == (row, col)

    def is_dead(self, row, col):
        """
        Check if the snake would die if it moved to the given cell
        :param row: Row of the head
        :param col: Column of the head
        :return: True if the snake has died, False otherwise
        """
        # If the next position of head is tail, snake lives.
        if (row, col) == self.tail.pos:     return False
        return self._off_grid(row, col) or self.is_snake(row, col) or self.grid[row, col] == SnakeState.OBSTACLE

    def _off_grid(self, row, col):
        """
        Check if the given cell is off the grid
        :param row: Row of the cell
        :param col: Column of the cell
        :return: True if the cell is off the grid, False otherwise
        """
        return row < 0 or row >= self.grid.shape[0] or col < 0 or col >= self.grid.shape[1]

    def move(self, new_head, eat_food=False, eat_anti_food=False):
        """
        Move the snake to the new head position
        Logics are implemented in environment class and this method is only used to update the grid and states
        :param new_head: New head node
        :param eat_food: Whether the snake eats food
        """
        # Link new head node in front of the current head
        new_head + self.head
        self.head = new_head

        eat_poison_food = self.config.poison and eat_anti_food
        eat_food = (not eat_poison_food) and (eat_food or eat_anti_food)

        # Destroy the tail node
        if not eat_food:
            self.grid[self.tail.pos] = SnakeState.EMPTY
            self.tail.destroy()
            self.tail = self.tail.prev_node

        # Update grid cell value
        self.grid[new_head.pos] = self.head.direction

        # update head previous node
        self.head.next_node.direction = self.head.direction
        self.grid[self.head.next_node.pos] = self.head.direction

    def remove_apple_node(self, position, eat_food, eat_anti_food):
        cursor = self._find_position_apple(position)
        if cursor == None:  return

        eat_poison_food = self.config.poison and eat_anti_food
        if eat_poison_food:
            self.poison_count = self.poison_count_max
            self.poisoned = True

        self.grid[cursor.pos] = SnakeState.EMPTY
        if self.anti_food_check:    self.grid[cursor.anti_pos] = SnakeState.EMPTY

        if cursor.next_node is None:
            cursor.prev_node.next_node = None
            self.last_food = cursor.prev_node
        elif cursor.prev_node is None:
            cursor.next_node.prev_node = None
            self.first_food = cursor.next_node
        else:
            cursor.prev_node.next_node = cursor.next_node
            cursor.next_node.prev_node = cursor.prev_node

    def _find_position_apple(self, position):
        cursor = self.first_food

        while cursor is not None:
            if cursor.pos == position and self.config.portal:
                self.next_portal_position = cursor.anti_pos
                return cursor
            elif cursor.anti_pos == position and self.config.portal:
                self.next_portal_position = cursor.pos
                return cursor
            elif cursor.pos == position or (self.anti_food_check and cursor.anti_pos == position):
                return cursor
            cursor = cursor.next_node
        return None

    def next_head_position(self, action):
        if self.poisoned:
            self.poison_count -= 1
            if self.poison_count == 0:
                self.poisoned = False
            action = 1

        if self.next_portal_position is not None:
            new_head_pos = self.next_portal_position
            self.next_portal_position = None
            return new_head_pos, SnakeAction.absolute_direction(self.head.direction, action)
        else:
            new_head_pos = SnakeAction.absolute_position(self.head.direction, action)(*self.head.pos)
            return new_head_pos, SnakeAction.absolute_direction(self.head.direction, action)


class SnakeObservation(Dict):
    """
    Observation space for the snake environment
    """

    def __init__(self, shape, multi_channel=False, direction_channel=False, dtype=np.uint8):
        """
        :param shape: Shape of the observation grid
        :param multi_channel: If True, the observation grid will be a 3-channel one-hot grid
        :param direction_channel: If False, treat snake directions as the same, else expose internal direction info
        """
        self.encodings = {
            SnakeState.EMPTY: 0,
            **{enum: 1 for enum in SnakeState if enum.name.startswith('SNAKE')},
            SnakeState.FOOD: 2,
            SnakeState.OBSTACLE: 3,
            SnakeState.ANTI_FOOD: 4,
        } if not direction_channel else {
            SnakeState.EMPTY: 0,
            **{enum: i + 1 for i, enum in enumerate(SnakeState) if enum.name.startswith('SNAKE')},
            SnakeState.FOOD: 5,
            SnakeState.OBSTACLE: 6,
            SnakeState.ANTI_FOOD: 7,
        }

        # -1 for binary representation except SnakeState.EMPTY
        self.n_states = SnakeState.n_states_internal() if direction_channel else SnakeState.n_states_external()
        self.n_channels = 1 if not multi_channel else self.n_states - 1
        space_dict = {}
        if multi_channel:
            shape = (self.n_channels, *shape)
            space_dict.update({'grid': Box(low=0, high=1, shape=shape, dtype=dtype)})
        else:
            shape = (1, *shape)
            space_dict.update({'grid': Box(low=0, high=self.n_states - 1, shape=shape, dtype=dtype)})
        space_dict.update(OrderedDict({
            'head_row': OneHotBox(shape[1]),
            'head_col': OneHotBox(shape[2]),
            'head_direction': OneHotBox(4),
            'tail_row': OneHotBox(shape[1]),
            'tail_col': OneHotBox(shape[2]),
            'tail_direction': OneHotBox(4),
            # 'portal_row': OneHotBox(shape[1]+1),
            # 'portal_col': OneHotBox(shape[2]+1)
        }))
        super(SnakeObservation, self).__init__(space_dict)
        self.dtype = dtype

    def encode(self, obs):
        """
        Encode the internal states to the simplified observation space
        Result is irreversible
        :param obs: np.ndarray of shape (row, col) containing the internal states
        :return: np.ndarray of shape (row, col) containing the observation space
        """
        return np.vectorize(self.encodings.get)(obs).astype(self.dtype)

    def convert(self, state: SnakeGrid):
        """
        Convert the internal states to the simplified observation space
        :param state: SnakeState object representing the internal states
        :return: np.ndarray of shape (n_channels, row, col) representing the observation space
        """
        obs_dict = OrderedDict()
        obs = state.grid
        if self.n_channels == 1:
            # Unsqueeze to add the channel dimension
            obs = self.encode(obs)[None, ...]
        else:
            # One-hot encoding
            # obs.shape = (H, W)
            # self.encode(obs).shape = (H, W) (with encoded labels)
            # ohe.shape = (H, W, n_states) (order is the same as self.encodings)
            ohe = np.eye(self.n_states, dtype=self.dtype)[self.encode(obs)]
            # Cut off the first channel containing SnakeState.EMPTY
            obs = ohe.transpose((-1, *list(range(len(ohe.shape))[:-1])))[1:]

        obs_dict.update({
            'grid': obs,
            'head_row': np.eye(state.grid.shape[0], dtype=self.dtype)[state.head.pos[0]],
            'head_col': np.eye(state.grid.shape[1], dtype=self.dtype)[state.head.pos[1]],
            'head_direction': np.eye(4, dtype=self.dtype)[state.head.direction - min(SnakeState)],
            'tail_row': np.eye(state.grid.shape[0], dtype=self.dtype)[state.tail.pos[0]],
            'tail_col': np.eye(state.grid.shape[1], dtype=self.dtype)[state.tail.pos[1]],
            'tail_direction': np.eye(4, dtype=self.dtype)[state.tail.direction - min(SnakeState)],
            # 'portal_row': np.eye(state.grid.shape[0] + 1, dtype=self.dtype)[0] if (
            #             state.next_portal_position is None) else \
            #     np.eye(state.grid.shape[0] + 1, dtype=self.dtype)[state.next_portal_position[0]],
            # 'portal_col': np.eye(state.grid.shape[1] + 1, dtype=self.dtype)[0] if (
            #             state.next_portal_position is None) else \
            #     np.eye(state.grid.shape[1] + 1, dtype=self.dtype)[state.next_portal_position[1]]
        })
        assert self['grid'].contains(obs_dict['grid']), f'grid does not match with the observation space'
        assert self['head_row'].contains(obs_dict['head_row']), f'head_row does not match with the observation space'
        assert self['head_col'].contains(obs_dict['head_col']), f'head_col does not match with the observation space'
        assert self['head_direction'].contains(
            obs_dict['head_direction']), f'direction does not match with the observation space'
        assert self['tail_row'].contains(obs_dict['tail_row']), f'head_row does not match with the observation space'
        assert self['tail_col'].contains(obs_dict['tail_col']), f'head_col does not match with the observation space'
        assert self['tail_direction'].contains(obs_dict['tail_direction']), f'direction does not match with the observation space'
        # assert self['portal_row'].contains(obs_dict['portal_row']), f'direction does not match with the observation space'
        # assert self['portal_col'].contains(obs_dict['portal_col']), f'direction does not match with the observation space'
        assert self.contains(obs_dict), f'Observation does not match with the observation space'

        return obs_dict

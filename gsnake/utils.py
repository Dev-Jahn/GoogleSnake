from enum import IntEnum, unique, auto

import numpy as np


@unique
class SnakeState(IntEnum):
    """
    Enum representing the state of a cell in the grid
    May add more states in the future
    Start from 1 for compatibility
    """
    SNAKE_U = 1
    SNAKE_R = auto()
    SNAKE_D = auto()
    SNAKE_L = auto()
    EMPTY = auto()
    FOOD = auto()
    OBSTACLE = auto()

    def __str__(self):
        return f'{self.name}<{self.value}>'
    def __repr__(self):
        return f'{self.name}<{self.value}>'



@unique
class SnakeAction(IntEnum):
    """
    Enum representing the action of the snake
    """
    LEFT = -1
    FORWARD = 0
    RIGHT = 1
    # Only for internal calculations
    BACKWARD = 2

    @staticmethod
    def absolute_direction(direction, action):
        """
        Get the direction of the snake after taking the given action
        :param direction: Direction of the snake before taking the action
        :param action: Action to take
        :return: Direction of the snake after taking the action
        """
        # return (direction + action) % 4
        return (direction + action-1) % 4 + 1  # gui compat

    @staticmethod
    def absolute_position(direction, action):
        """
        Get the lambda function to calculate absolute position of the snake after taking an action
        :param direction: The direction of the snake
        :param action: The action to take
        :return: The lambda function takes row, col as input and relative position as output
        """
        print('DIR:', direction)
        print('ACT:', action)
        print('ABS DIR:', SnakeAction.absolute_direction(direction, action))
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


class SnakeGrid:
    """
    Grid class for the GoogleSnake game environment
    Extends the Box class from gym.Space and contains actual cell states of grid
    """

    def __init__(self, config, init_length=2, seed=None):
        self.config = config
        self.init_length = init_length
        self.seed = seed
        self.grid = np.zeros(config.grid_shape, dtype=np.uint8)
        self.reset()

        # Cursor of the head and tail SnakeNode
        self.head = None
        self.tail = None
        self.reset()

    def reset(self):
        """
        Reset the grid to all empty
        :return:
        """
        self.grid.fill(SnakeState.EMPTY)
        self._init_snake()
        self.generate_food(n=self.config.n_foods)

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

    def generate_food(self, n=1):
        """
        Generate food in an empty cell
        :return: None
        """
        coords = self._sample_empty(n=n)
        for coord in coords:
            self.grid[coord] = SnakeState.FOOD

    def generate_obstacles(self, n=1):
        """
        Generate obstacles in empty cells
        :return: None
        """
        self.grid[self._sample_empty(n)] = SnakeState.OBSTACLE

    def _sample_empty(self, n=1):
        """
        Sample a random empty cell from the grid
        :return: Tuples of ((row1, col1), (row2, col2), ...)
        """
        empty = np.argwhere(self.grid == SnakeState.EMPTY)
        return tuple(tuple(row) for row in empty[np.random.randint(len(empty), size=n), :])

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
        return self._off_grid(row, col) or self.is_snake(row, col) or self.grid[row, col] == SnakeState.OBSTACLE

    def _off_grid(self, row, col):
        """
        Check if the given cell is off the grid
        :param row: Row of the cell
        :param col: Column of the cell
        :return: True if the cell is off the grid, False otherwise
        """
        return row < 0 or row >= self.grid.shape[0] or col < 0 or col >= self.grid.shape[1]

    def move(self, new_head, eat_food=False, portal=False):
        """
        Move the snake to the new head position
        Logics are implemented in environment class and this method is only used to update the grid and states
        :param new_head: New head node
        :param eat_food: Whether the snake eats food
        """
        # Link new head node in front of the current head
        new_head + self.head
        self.head = new_head

        # Destroy the tail node
        if not eat_food:
            self.grid[self.tail.pos] = SnakeState.EMPTY
            self.tail.destroy()
            self.tail = self.tail.prev_node

        # Update grid cell value
        self.grid[new_head.pos] = self.head.direction

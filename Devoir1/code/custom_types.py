from enum import Enum
from typing import Literal
from game import Directions

class DirectionEnum(Enum):
    NORTH = Directions.NORTH
    SOUTH = Directions.SOUTH
    EAST = Directions.EAST
    WEST = Directions.WEST
    STOP = Directions.STOP

Direction = Literal[DirectionEnum.EAST,
                 DirectionEnum.WEST,
                 DirectionEnum.NORTH,
                 DirectionEnum.SOUTH,
                 DirectionEnum.STOP]

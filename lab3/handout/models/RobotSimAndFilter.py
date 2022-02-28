
import random
from typing import Tuple
import numpy as np

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#
class RobotSim:
    def __init__(self, rows, cols):
        self.__rows = rows
        self.__cols = cols

        # Initialize with random vaild position and heading
        self.position = (random.randint(0, cols-1), random.randint(0, rows-1))
        self.heading = random.randint(0, 3)

    def move(self):
        if self.__is_encountering_wall():
            while self.__is_encountering_wall():
                self.__rotate()
        if random.random() < 0.3:
            self.__rotate()
            while self.__is_encountering_wall():
                self.__rotate()
        
        self.position = self.__move()

    def read_sensor(self):
        p_true_position = 0.1
        p_one_position_off = 0.05
        p_two_positions_off = 0.025

        one_off_positions = self.__get_surrounding_positions(max_distance=1)
        two_off_positions = self.__get_surrounding_positions(max_distance=2)

        p_return_one_position_off = p_one_position_off * len(one_off_positions)
        p_return_two_positions_off = p_two_positions_off * len(two_off_positions)

        sample = random.random()

        # Select which kind of position to return based on which interval the sample is in
        if sample <= p_true_position:
            return self.position
        elif sample < (p_true_position + p_return_one_position_off):
            return random.choice(one_off_positions)
        elif sample < (p_true_position + p_return_one_position_off + p_return_two_positions_off):
            return random.choice(two_off_positions)
        else:
            return None

    def __is_encountering_wall(self):
        x_next, y_next = self.__move()
        return x_next < 0 \
            or self.__rows <= y_next \
            or y_next < 0  \
            or self.__cols <= x_next

    def __get_surrounding_positions(self, max_distance) -> list[Tuple[int, int]]:
        (x_current, y_current) = self.position

        surrounding_positions = []
        for x in range(x_current - max_distance, x_current + max_distance):
            for y in range(y_current - max_distance, y_current + max_distance):
                if x in list(range(0, self.__cols)) and y in list(range(0, self.__rows)):
                    surrounding_positions.append((x, y))
        
        return surrounding_positions

    def __rotate(self):
        self.heading = random.randint(0,3)

    def __move(self):
        (x, y) = self.position
        new_coordinates = {
            0: (x, y + 1), #NORTH
            1: (x + 1, y), #EAST
            2: (x, y - 1), #SOUTH
            3: (x - 1, y) #WEST
        }
        return new_coordinates[self.heading]


            
#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:

    def __init__(self, observation_model: ObservationModel, transition_model: TransitionModel, state_model: StateModel):
        self.__observation_model = observation_model
        self.__state_model = state_model
        self.__transition_model = transition_model
        self.__f_matrix = self.__init_f_matrix() # Uniform priors

    def __init_f_matrix(self):
        rows, cols, headings = self.__state_model.get_grid_dimensions()
        length = rows * cols * headings
        return np.repeat([float(1) / length], length)
        
    def predict_position(self, reading: int) -> Tuple[int, int]:
        O = self.__observation_model.get_o_reading(reading)
        f = self.__f_matrix
        t = self.__transition_model

        self.__f_matrix = O.dot(t.get_T_transp()).dot(f)
        self.__f_matrix /= np.sum(self.__f_matrix)

        max_prob_idx = np.argmax(self.__f_matrix)
        rows, cols, headings = self.__state_model.get_grid_dimensions()
        x = (max_prob_idx // 4) // rows
        y = (max_prob_idx // 4) % rows
        return (x, y), self.__f_matrix

        
        
        

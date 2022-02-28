
import random
import numpy as np

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#
class RobotSim:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        # Initialize with random vaild position and heading
        self.position = (random.randint(0, cols-1), random.randint(0, rows-1))
        self.heading = random.randint(0, 3)

    def move(self):
        print('Robot moving')
        if self.__is_encountering_wall():
            while self.__is_encountering_wall():
                self.__rotate()
        elif random.random() < 0.3:
            self.__rotate()
        
        self.position = self.__move()
        print('New Position:', str(self.position))

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
        elif p_true_position <= sample < (p_true_position + p_return_one_position_off):
            return random.choice(one_off_positions)
        elif (p_true_position + p_return_one_position_off) <= sample < (p_true_position + p_return_one_position_off + p_return_two_positions_off):
            return random.choice(two_off_positions)
        else:
            return None

    def __is_encountering_wall(self):
        return (self.position[0] == 0 and self.heading == 3) \
            or (self.position[0] == self.cols and self.heading == 1) \
            or (self.position[1] == 0 and self.heading == 0) \
            or (self.position[1] == self.rows and self.heading == 2)

    def __get_surrounding_positions(self, max_distance):
        (x_current, y_current) = self.position

        surrounding_positions = []
        for x in range(x_current - max_distance, x_current + max_distance):
            for y in range(y_current - max_distance, y_current + max_distance):
                if x in list(range(0, self.cols)) and y in list(range(0, self.rows)):
                    surrounding_positions.append((x, y))
        
        return surrounding_positions

    def __rotate(self):
        self.heading = random.randint(0,3)
        print("Robot changed heading", self.heading)

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
    def __init__(self, rows, cols):
        print("Hello again, World")

        
        
        

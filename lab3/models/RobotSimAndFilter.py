
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

        self.position = random.randint(0, cols - 1), random.randint(0, rows - 1)
        self.heading = random.randint(0, 3)

    # Makes a move based on current heading and position (with 30% chance of changing direction)
    def move(self): 
        print("Robot moving")
        if self.is_facing_wall():
            while self.is_facing_wall():
                self.change_direction()
        elif random.random() <= 0.3: 
            self.change_direction()

        self.position = self.simulate_move_forward()
        print("Robot new position" + str(self.position))

    # Returns the position based on sensor reading.
    def sense(self):
        p_true_pos = 0.1
        p_n_Ls = 0.05
        p_n_Ls2 = 0.025
        
        rand = random.random()

        if rand <= p_true_pos: # True location
            return self.position
        elif rand <= p_n_Ls * 8: # First circle
            return self._get_first_surrounding_location()
        elif rand <= p_n_Ls2 * 16: # Second circle
            return self._get_second_surrounding_location()
        else: # No reading
            return None

    # --- Helpers ---

    def change_direction(self): 
        self.heading = random.randint(0,3)
        print("Robot: changed direction", self.heading)

    def simulate_move_forward(self):
        (x, y) = self.position

        # North, east, south, west
        moves = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        next_move = moves[self.heading]
        return next_move

    def is_facing_wall(self):
        x_straight, y_straight = self.simulate_move_forward()
        return not ( 0 < x_straight <= self.cols or 0 < y_straight <= self.rows )

    def _get_first_surrounding_location(self):
        x, y = self.position
        possible_surrounding_position = [[x + i, y + j] for i, j in zip([-1,-1,-1, 0,0, 1,1,1], [-1,0,1, -1,1, -1,0,1])]
        chosen_point = random.choice(possible_surrounding_position)
        (x, y) = chosen_point

        if ( 0 < x <= self.cols or 0 < y <= self.rows ):
            return None
        else:
            return chosen_point

    def _get_second_surrounding_location(self):
        x, y = self.position
        possible_surrounding_position = [[x + i, y + j] for i, j in zip([-2,-2,-2,-2,-2, -1,-1, 0,0, 1,1, 2,2,2,2,2], [-2,-1,0,1,2, -2,2, -2,2, -2,2, -2,-1,0,1,2])]
        chosen_point = random.choice(possible_surrounding_position)
        (x, y) = chosen_point

        if ( 0 < x <= self.cols or 0 < y <= self.rows ):
            return None
        else:
            return chosen_point

        
#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Initialize f with uniform probability
        possible_moves = self.width * self.height * 4
        f = np.repeat(1/possible_moves, possible_moves)

    def forward(self, t, sense):
        o = self.create_sensor_matrix(sense)
        self.f = o.dot(np.transpose(t)).dot(self.f)
        
        return self.f

    def predict_position(self):
        max_prob_idx = np.argmax(self.f)
        x = (max_prob_idx // 4) // self.height
        y = (max_prob_idx // 4) % self.height
        return (x, y)

    # @TODO: Change mehtod!!!!
    def create_sensor_matrix(self, sensed_coord):
        if sensed_coord is None:
            return self.none_matrix
        width = self.width
        height = self.height
        o = np.array(np.zeros(shape=(width * height * 4, width * height * 4)))
        x, y = sensed_coord

        # prob of 0.1
        index = x * 4 * height + y * 4
        for i in range(4):
            o[index + i, index + i] = 0.1

        # prob 0.05
        self.assign_adj(o, self.possible_adj(x, y), 0.05)
        # prob 0.025
        self.assign_adj(o, self.possible_adj2(x, y), 0.025)
        
        
        


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

        self.position = random.randint(0, rows - 1), random.randint(0, cols - 1)
        self.heading = random.randint(0, 3)

    # Makes a move based on current heading and position (with 30% chance of changing direction)
    def move(self): 
        if random.random() <= 0.3: 
            self.change_direction()
        if self.is_facing_wall():
            self.change_direction()
            while self.is_facing_wall():
                self.change_direction()

        self.position = self.simulate_move_forward()
        return (self.position[0], self.position[1], self.heading)

    # Returns the position based on sensor reading.
    def sense(self):
        p_true_pos = 0.1
        p_n_Ls = 0.05
        p_n_Ls2 = 0.025
        
        rand = random.random()
        if rand <= p_true_pos: # True location
            res = self.position
        elif rand <= p_true_pos + p_n_Ls * 8: # First circle
            res = self._get_first_surrounding_location()
        elif rand <= p_true_pos + p_n_Ls * 8 + p_n_Ls2 * 16: # Second circle
            res = self._get_second_surrounding_location()
        else: # No reading
            res = None
        
        # print("Robot sensed", res)
        return res

    # --- Helpers ---

    def change_direction(self): 
        self.heading = random.randint(0,3)
        # print("Robot: changed direction", self.heading)

    def simulate_move_forward(self):
        (x, y) = self.position

        # North, east, south, west
        moves = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        next_move = moves[self.heading]
        return next_move

    def is_facing_wall(self):
        x_straight, y_straight = self.simulate_move_forward()
        return ( x_straight < 0  or self.rows <= x_straight or y_straight < 0  or self.cols <= y_straight )

    def _get_first_surrounding_location(self):
        x, y = self.position
        possible_surrounding_position = [[x + i, y + j] for i, j in zip([-1,-1,-1, 0,0, 1,1,1], [-1,0,1, -1,1, -1,0,1])]
        chosen_point = random.choice(possible_surrounding_position)
        (x, y) = chosen_point

        if ( x < 0  or self.rows <= x or y < 0  or self.cols <= y ):
            # print("(1st circle) Chose WALL point", x, y)
            return None
        else:
            # print("(1st circle) Chose point", x, y, self.rows, self.cols)
            return chosen_point

    def _get_second_surrounding_location(self):
        x, y = self.position
        possible_surrounding_position = [[x + i, y + j] for i, j in zip([-2,-2,-2,-2,-2, -1,-1, 0,0, 1,1, 2,2,2,2,2], [-2,-1,0,1,2, -2,2, -2,2, -2,2, -2,-1,0,1,2])]
        chosen_point = random.choice(possible_surrounding_position)
        (x, y) = chosen_point

        if ( x < 0  or self.rows <= x or y < 0  or self.cols <= y ):
            # print("(2nd sense) Chose WALL point", x, y)
            return None
        else:
            # print("(2nd sense) Chose point", x, y, self.rows, self.cols)
            return chosen_point

        
#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:
    def __init__(self, width, height, sm):
        self.width = width
        self.height = height
        self.__sm = sm

        # Initialize f with uniform probability
        possible_moves = self.width * self.height * 4

        # Normalize f
        self.f = np.repeat(1/possible_moves, possible_moves)

    def forward(self, t_transpose, o):
        self.f = o.dot(t_transpose).dot(self.f)
        self.f /= np.sum(self.f)
        
        return self.f

    def predict_position(self):
        max_prob_idx = np.argmax(self.f)
        return self.__sm.reading_to_position(max_prob_idx // 4)
        
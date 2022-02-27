
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
            if is_facing_wall():
                while is_facing_wall():
                    self.heading = self.get_random_direction()
            elif random.random() <= 0.3: 
                self.heading = self.get_random_direction()

            self.position = simulate_move_forward()

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

        def get_random_direction(self): 
            return random.randint(0,3)

        def simulate_move_forward():
            (x, y) = self.position

            # North, east, south, west
            moves = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
            next_move = moves[self.heading]
            return next_move

        def is_facing_wall(self):
            x_straight, y_straight = self.simulate_move_forward()
            return ( 0 < x_straight <= self.width or 0 < y_straight <= self.height )

        def _get_first_surrounding_location(self):
            x, y = self.position
            possible_surrounding_position = [[x + i, y + j] for i, j in zip([-1,-1,-1, 0,0, 1,1,1], [-1,0,1, -1,1, -1,0,1])]
            chosen_point = random.choice(possible_surrounding_position)
            (x, y) = chosen_point

            if ( 0 < x <= self.width or 0 < y <= self.height ):
                return None
            else:
                return chosen_point

        def _get_second_surrounding_location(self):
            x, y = self.position
            possible_surrounding_position = [[x + i, y + j] for i, j in zip([-2,-2,-2,-2,-2, -1,-1, 0,0, 1,1, 2,2,2,2,2], [-2,-1,0,1,2, -2,2, -2,2, -2,2, -2,-1,0,1,2])]
            chosen_point = random.choice(possible_surrounding_position)
            (x, y) = chosen_point

            if ( 0 < x <= self.width or 0 < y <= self.height ):
                return None
            else:
                return chosen_point

        
#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:
    def __init__(self):
        print("Hello again, World")

        
        
        

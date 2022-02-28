
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
#

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import random

from models import StateModel,TransitionModel,ObservationModel,RobotSimAndFilter

class Localizer:
    def __init__(self, state_model: StateModel):

        self.__state_model = state_model

        self.__transition_model = TransitionModel(self.__state_model)
        self.__observation_model = ObservationModel(self.__state_model)

        # change in initialise in case you want to start out with something else
        # initialise can also be called again, if the filtering is to be reinitialised without a change in size
        self.initialise()

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.ndarray:
        return self.__transition_model

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.ndarray:
        return self.__observation_model

    # the current true pose (x, h, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> Tuple[int, int, int]:
        x, y, heading = self.__state_model.state_to_pose(self.__trueState)
        return x, y, heading

    # the current probability distribution over all states
    def get_current_f_vector(self) -> np.array:
        return self.__probs

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> Tuple[int, int]:
        ret = None
        if self.__sense != None:
            ret = self.__state_model.reading_to_position(self.__sense)
        return ret;

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> Tuple[int, int]:
        return self.__estimate

    ################################### Here you need to really fill in stuff! ##################################
    # if you want to start with something else, change the initialisation here!
    #
    # (re-)initialise for a new run without change of size
    def initialise(self):
        self.__trueState = random.randint(0, self.__state_model.get_num_of_states() - 1)
        self.__sense = None
        self.__probs = np.ones(self.__state_model.get_num_of_states()) / (self.__state_model.get_num_of_states())
        self.__estimate = self.__state_model.state_to_position(np.argmax(self.__probs))
    
    # add your simulator and filter here, for example    
        
        self.__robot_simulator = RobotSimAndFilter.RobotSim(self.__state_model.get_grid_dimensions()[0], self.__state_model.get_grid_dimensions()[1])
        self.__HMM = RobotSimAndFilter.HMMFilter(self.__observation_model, self.__transition_model, self.__state_model)
    #
    #  Implement the update cycle:
    #  - robot moves one step, generates new state / pose
    #  - sensor produces one reading based on the true state / pose
    #  - filtering approach produces new probability distribution based on
    #  sensor reading, transition and sensor models
    #
    #  Add an evaluation in terms of Manhattan distance (average over time) and "hit rate"
    #  you can do that here or in the simulation method of the visualisation, using also the
    #  options of the dashboard to show errors...
    #
    #  Report back to the caller (viewer):
    #  Return
    #  - true if sensor reading was not "nothing", else false,
    #  - AND the three values for the (new) true pose (x, y, h),
    #  - AND the two values for the (current) sensor reading (if not "nothing")
    #  - AND the error made in this step
    #  - AND the new probability distribution
    #
    def update(self, verbose=False) -> Tuple[bool, int, int, int, int, int, int, int, int, np.array] :
        # update all the values to something sensible instead of just reading the old values...
        # 
        self.__robot_simulator.move()
        self.__trueState = self.__state_model.pose_to_state(self.__robot_simulator.position[0], self.__robot_simulator.position[1], self.__robot_simulator.heading)
        if verbose: print('Robot moved. New position:', str(self.__robot_simulator.position))

        sense_position = self.__robot_simulator.read_sensor()
        self.__sense = None if not sense_position else self.__state_model.position_to_reading(sense_position[0], sense_position[1])
        if verbose: print('Sensor detected position:', str(sense_position))

        self.__estimate, self.__probs = self.__HMM.predict_position(self.__sense)
        if verbose: print("Estimated position:", str(self.__estimate))

        # this block can be kept as is
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__state_model.state_to_pose(self.__trueState)
        srX = -1
        srY = -1
        if self.__sense != None:
            srX, srY = self.__state_model.reading_to_position(self.__sense)
            ret = True
            
        eX, eY = self.__estimate
        
        # this should be updated to spit out the actual error for this step
        error = abs(tsX - eX) + abs(tsY - eY)
        if verbose: print('#### ERROR:', error)                
        
        # if you use the visualisation (dashboard), this return statement needs to be kept the same
        # or the visualisation needs to be adapted (your own risk!)
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, self.__probs

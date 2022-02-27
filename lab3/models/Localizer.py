
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
#

import numpy as np
import matplotlib.pyplot as plt
import random

from models import StateModel,TransitionModel,ObservationModel,RobotSimAndFilter

class Localizer:
    __rs: RobotSimAndFilter.RobotSim
    __HMM: RobotSimAndFilter.HMMFilter
    __sm: StateModel

    def __init__(self, sm, rows, cols):

        self.__sm = sm

        self.__tm = TransitionModel(self.__sm)
        self.__om = ObservationModel(self.__sm)

        # change in initialise in case you want to start out with something else
        # initialise can also be called again, if the filtering is to be reinitialised without a change in size
        self.initialise(rows, cols)

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.array:
        return self.__tm

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.array:
        return self.__om

    # the current true pose (x, h, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> (int, int, int):
        x, y, h = self.__sm.state_to_pose(self.__trueState)
        return x, y, h

    # the current probability distribution over all states
    def get_current_f_vector(self) -> np.array(float):
        return self.__probs

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> (int, int):
        ret = None
        if self.__sense != None:
            ret = self.__sm.reading_to_position(self.__sense)
        return ret;

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> (int, int):
        return self.__estimate

    ################################### Here you need to really fill in stuff! ##################################
    # if you want to start with something else, change the initialisation here!
    #
    # (re-)initialise for a new run without change of size
    def initialise(self, rows, cols):
        self.__trueState = random.randint(0, self.__sm.get_num_of_states() - 1)
        self.__sense = None
        self.__probs = np.ones(self.__sm.get_num_of_states()) / (self.__sm.get_num_of_states())
        self.__estimate = self.__sm.state_to_position(np.argmax(self.__probs))
    
        # add your simulator and filter here, for example    
        self.__rs = RobotSimAndFilter.RobotSim(rows, cols)
        self.__HMM = RobotSimAndFilter.HMMFilter(rows, cols)

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
    def update(self) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        # update all the values to something sensible instead of just reading the old values...
        
        self.__rs.move()
        print("New real position" + str(self.__rs.position))
        
        self.__sense = self.__rs.sense()
        print("robot senses location: " + str(self.__sense))
        
        self.__f = self.__HMM.forward(self.__tm, self.__sense)
        
        self.__estimate = self.__HMM.predict_position()
        print("Predicted position: " + str(self.__estimate))        

        
        # this block can be kept as is
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__sm.state_to_pose(self.__trueState)
        srX = -1
        srY = -1
        if self.__sense != None:
            srX, srY = self.__sm.reading_to_position(self.__sense)
            ret = True
            
        eX, eY = self.__estimate
        
        # this should be updated to spit out the actual error for this step
        error = 10.0                
        
        # if you use the visualisation (dashboard), this return statement needs to be kept the same
        # or the visualisation needs to be adapted (your own risk!)
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, self.__probs

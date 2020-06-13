import random

import dm_env
import numpy as np
from acme import types, core
from acme.adders import Adder
from tensorflow import keras


class DQNActor(core.Actor):

    def __init__(self,
                 model: keras.Model,
                 adder: Adder,
                 starting_exploration_rate: float,
                 exploration_rate_decay: float):
        self.model = model
        self.outputs = self.model.layers[len(self.model.layers) - 1].output.shape[1]
        self.adder = adder
        self.exploration_rate = starting_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        rand = random.random()
        if rand < self.exploration_rate:
            action = np.random.randint(0, self.outputs)
        else:
            q_values = self.get_q_values(observation)
            action = np.argmax(q_values)
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(
            self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
    ):
        self.adder.add(action=action, next_timestep=next_timestep)

        if next_timestep.step_type == dm_env.StepType.LAST:
            self.exploration_rate = self.exploration_rate * self.exploration_rate_decay

    def update(self):
        pass

    def get_q_values(self, state: types.NestedArray):
        predicted = self.model.predict(state.reshape(1, len(state)))
        return predicted[0]



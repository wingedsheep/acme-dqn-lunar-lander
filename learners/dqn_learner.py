from datetime import datetime
from typing import Tuple

import numpy as np
from acme import core, types
from dm_env import StepType
from tensorflow import keras

from keras.keras_saveable import KerasSaveable
from objects.memory import Memory
from objects.memory_entry import MemoryEntry


class DQNLearner(core.Learner, KerasSaveable):

    def __init__(self,
                 memory: Memory,
                 network: keras.Model,
                 target_network: keras.Model,
                 input_shape: Tuple[int],
                 output_size: int,
                 batch_size: int = 256,
                 minimum_memory_size: int = 256,
                 discount_factor: float = 0.99):
        self.memory = memory
        self.network = network
        self.target_network = target_network
        self.input_shape = input_shape,
        self.output_size = output_size,
        self.batch_size = batch_size
        self.minimum_memory_size = minimum_memory_size
        self.discount_factor = discount_factor
        self.__steps = 0

    def step(self):
        if self.__steps != 0 and self.__steps % 10 == 0:
            self.learn_on_mini_batch()

        if self.__steps != 0 and self.__steps % 1000 == 0:
            name = "network_backups/backup_" + str(self.__steps) + "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.network.save(name)  # saves compiled state
            self.target_network = keras.models.load_model(name)

        self.__steps += 1

    def learn_on_mini_batch(self):
        if self.memory.num_entries() > self.minimum_memory_size:
            batch: [MemoryEntry] = self.memory.get_batch(self.batch_size)
            x_batch = []
            y_batch = []

            sample: MemoryEntry
            for sample in batch:
                observation = sample.prev_timestep.observation
                next_observation = sample.timestep.observation
                reward = sample.timestep.reward
                final = sample.timestep.step_type == StepType.LAST

                q_values = self.get_q_values(self.network, observation)
                q_values_new_state = self.get_q_values(self.target_network, observation)

                target_value = self.calculate_target(
                    q_values_new_state,
                    reward,
                    final
                )

                x_batch.append(observation)
                y_sample = q_values
                y_sample[sample.action] = target_value
                y_batch.append(y_sample)
                if final:
                    x_batch.append(next_observation)
                    y_batch.append([reward] * self.output_size[0])
            self.network.fit(np.array(x_batch), np.array(y_batch), batch_size=len(batch), epochs=1, verbose=0)

    @staticmethod
    def get_q_values(network: keras.Model, state: types.NestedArray):
        predicted = network.predict(state.reshape(1, len(state)))
        return predicted[0]

    @staticmethod
    def get_max_q(q_values_new_state):
        return np.max(q_values_new_state)

    def calculate_target(self, q_values_new_state, reward, final):
        if final:
            return reward
        else:
            return reward + self.discount_factor * self.get_max_q(q_values_new_state)

    def get_variables(self, names):
        return {
            ''
        }

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            # 'network': self._network,
            # 'target_network': self._target_network,
            # 'optimizer': self._optimizer,
            # 'num_steps': self._num_steps
        }

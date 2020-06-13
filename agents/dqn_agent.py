from acme import specs
from acme.agents import agent
from tensorflow import keras

from actors.dqn_actor import DQNActor
from adders.fifo_adder import FifoAdder
from learners.dqn_learner import DQNLearner

import tensorflow as tf

from objects.memory import Memory


class DQNAgent(agent.Agent):

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            network: keras.Model,
            batch_size: int = 256,
            prefetch_size: int = 4,
            target_update_period: int = 100,
            samples_per_insert: float = 32.0,
            min_replay_size: int = 1000,
            max_replay_size: int = 1000000,
            importance_sampling_exponent: float = 0.2,
            priority_exponent: float = 0.6,
            n_step: int = 5,
            epsilon: tf.Tensor = None,
            learning_rate: float = 1e-3,
            discount: float = 0.99,
    ):
        self.network = network
        self.network.compile(loss="categorical_crossentropy", optimizer='adam', learning_rate=learning_rate)

        name = "network_backup"
        self.network.save(name)  # saves compiled state
        self.target_network = keras.models.load_model(name)

        memory = Memory(
            size=max_replay_size
        )

        adder = FifoAdder(
            memory=memory
        )

        actor = DQNActor(
            model=network,
            adder=adder,
            starting_exploration_rate=1.0,
            exploration_rate_decay=0.995
        )

        learner = DQNLearner(
            memory=memory,
            network=self.network,
            target_network=self.target_network,
            input_shape=environment_spec.observations.shape,
            output_size=environment_spec.actions.num_values,
            batch_size=128,
            minimum_memory_size=128,
            discount_factor=discount
        )

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=1)

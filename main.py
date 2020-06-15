import acme
import gym
from acme import wrappers, specs

from acme.utils import loggers
from acme.wrappers import gym_wrapper

from agents.dqn_agent import DQNAgent
from networks.models import Models

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


def render(env):
    return env.environment.render(mode='rgb_array')


environment = gym_wrapper.GymWrapper(gym.make('LunarLander-v2'))
environment = wrappers.SinglePrecisionWrapper(environment)
environment_spec = specs.make_environment_spec(environment)

model = Models.sequential_model(
    input_shape=environment_spec.observations.shape,
    num_outputs=environment_spec.actions.num_values,
    hidden_layers=3,
    layer_size=300
)

agent = DQNAgent(
    environment_spec=environment_spec,
    network=model
)

logger = loggers.TerminalLogger(time_delta=10.)
loop = acme.EnvironmentLoop(
    environment=environment,
    actor=agent
)
loop.run()

import abc
import dm_env
from acme import types
from acme.adders import Adder

from objects.memory import Memory
from objects.memory_entry import MemoryEntry


class FifoAdder(Adder):

    def __init__(self, memory: Memory):
        self.memory = memory
        self.prev_timestep = None

    def add_first(self, timestep: dm_env.TimeStep):
        """Defines the interface for an adder's `add_first` method.

        We expect this to be called at the beginning of each episode and it will
        start a trajectory to be added to replay with an initial observation.

        Args:
          timestep: a dm_env TimeStep corresponding to the first step.
        """
        self.prev_timestep = timestep

    def add(
            self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
            extras: types.NestedArray = (),
    ):
        """Defines the adder `add` interface.

        Args:
          action: A possibly nested structure corresponding to a_t.
          next_timestep: A dm_env Timestep object corresponding to the resulting
            data obtained by taking the given action.
          extras: A possibly nested structure of extra data to add to replay.
        """
        if self.prev_timestep is not None:
            self.memory.add_memory(
                MemoryEntry(
                    action=action,
                    prev_timestep=self.prev_timestep,
                    timestep=next_timestep
                )
            )
        self.prev_timestep = next_timestep

    def reset(self):
        """Resets the adder's buffer."""
        self.prev_timestep = None

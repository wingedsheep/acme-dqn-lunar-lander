from typing import Optional

import dm_env
from acme import types


class MemoryEntry:

    def __init__(self, action: types.NestedArray, prev_timestep: Optional[dm_env.TimeStep], timestep: dm_env.TimeStep):
        self.action = action
        self.prev_timestep = prev_timestep
        self.timestep = timestep

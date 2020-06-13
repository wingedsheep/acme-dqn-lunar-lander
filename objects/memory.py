import random
from objects.memory_entry import MemoryEntry


class Memory:

    def __init__(self, size: int = 10000):
        self.size: int = size
        self.currentPosition: int = 0
        self.entries: [MemoryEntry] = []

    def get_batch(self, size) -> [MemoryEntry]:
        return random.sample(self.entries, size)

    def num_entries(self) -> int:
        return len(self.entries)

    def get_memory(self, index) -> MemoryEntry:
        return self.entries[index]

    def add_memory(self, entry: MemoryEntry):
        if self.currentPosition >= self.size - 1:
            self.currentPosition = 0
        if len(self.entries) > self.size:
            self.entries[self.currentPosition] = entry
        else:
            self.entries.append(entry)

        self.currentPosition += 1

    def reset(self):
        self.entries = []
        self.currentPosition = 0

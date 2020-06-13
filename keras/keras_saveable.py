import abc


class KerasSaveable(abc.ABC):

    @property
    @abc.abstractmethod
    def state(self):
        pass

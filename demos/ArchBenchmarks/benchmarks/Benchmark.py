from abc import ABC, abstractmethod

class Benchmark(ABC):

    @abstractmethod
    def run():
        raise NotImplementedError
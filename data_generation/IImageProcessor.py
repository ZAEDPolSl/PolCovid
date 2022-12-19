from abc import ABC, abstractmethod


class IImageProcessor(ABC):

    @abstractmethod
    def process_image(self):
        pass

    @abstractmethod
    def normalize(self):
        pass
    
    @abstractmethod
    def log(self):
        pass
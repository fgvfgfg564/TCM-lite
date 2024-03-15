from coding_tools.traditional_tools import WebPTool
import abc


class CodecBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, input_pth, output_pth, **kwargs):
        """
        Encode
        """

    @abc.abstractmethod
    def decode(self, input_pth):
        """
        Decode
        """

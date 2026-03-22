"""
All device related stuff.
This sctipt doesn't do much as of now.
When I add GPU support, this will make things easier.
"""

from typing import Literal, get_args

SupportedDevices = Literal["cpu"]


class Device:
    """
    Simple API to hold device objects.
    The Device instance is responsible for moving
    data between devices.
    """

    def __init__(self, name: SupportedDevices):
        assert name in get_args(SupportedDevices), (
            f"Unsupported device {name}, expected {get_args(SupportedDevices)}"
        )
        self.device = name

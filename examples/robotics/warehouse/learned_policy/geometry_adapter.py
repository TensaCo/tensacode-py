"""Same shared locomotion/servos, with the geometry-residual learned runtime."""

from .adapter import LearnedController
from .geometry_model import GeometryRuntime


class GeometryController(LearnedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, runtime_factory=GeometryRuntime, **kwargs)

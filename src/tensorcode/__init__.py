"""Composable cognitive operations, tools, and explicit execution traces."""
from ._internal.tracing import Trace, trace, InputRef, OutputRef

__version__ = "0.4.0a4"
__all__ = ["Trace", "trace", "InputRef", "OutputRef"]

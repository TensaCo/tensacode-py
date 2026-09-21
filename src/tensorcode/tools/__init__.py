"""Complete trainable tools; heavyweight dependencies load on first use."""
from importlib import import_module

__all__ = ['Chatbot', 'Investigator', 'Planner', 'Decision', 'Scene']


def __getattr__(name):
    modules = {'Chatbot': 'chatbot', 'Investigator': 'investigator',
               'Planner': 'planner', 'Decision': 'decision', 'Scene': 'scene'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f'{__name__}.{modules[name]}'), name)
    globals()[name] = value
    return value

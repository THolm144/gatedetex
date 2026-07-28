"""World-module discovery for gatedetex.

See ``gatedetex.worlds.registry`` for details, and ``_template.py`` for the
contract a world module must implement.
"""

from .registry import add_world_dir, list_worlds, load_world

__all__ = ["add_world_dir", "list_worlds", "load_world"]

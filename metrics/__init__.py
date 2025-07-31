"""
metrics.registry: lightweight plugin registry.

Add a new metric by creating a module inside `metrics/` that implements
`class Metric` with:

    name: str
    __call__(self, x, y) → torch.Tensor        # shape=(B,)
"""
from importlib import import_module
from pathlib import Path
from typing import Dict, List

__all__ = ["registry"]

class _Registry(dict):
    def register(self, modname: str) -> None:
        module = import_module(f"metrics.{modname}")
        self[module.Metric.name] = module.Metric

    def build(self, names: List[str], **kwargs):
        return [self[n](**kwargs) for n in names if n in self]

# discover sub-modules -----------------------------------------------------------
registry: _Registry = _Registry()
for p in Path(__file__).parent.glob("[!_]*.py"):
    registry.register(p.stem)

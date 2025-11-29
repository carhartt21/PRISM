"""
PRISM: Pipeline for Robust Image Similarity Metrics

A comprehensive evaluation toolkit for comparing generated images against
original reference images to assess the quality and realism of image-to-image translation
or weather synthesis models.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .utils import load_and_pair_images, summarise_metrics, configure_logger

__all__ = [
    "load_and_pair_images",
    "summarise_metrics", 
    "configure_logger",
    "registry"
]


def __getattr__(name):
    if name == "registry":
        from .metrics import registry as metric_registry

        return metric_registry
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

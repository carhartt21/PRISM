"""
Image Evaluation Pipeline

A comprehensive Python evaluation pipeline for comparing generated images 
against real images to assess the quality and realism of image-to-image 
translation or weather synthesis models.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .utils import load_and_pair_images, summarise_metrics, configure_logger
from .metrics import registry

__all__ = [
    "load_and_pair_images",
    "summarise_metrics", 
    "configure_logger",
    "registry"
]

"""
Evaluation module initialization
"""

from .metrics import ModelEvaluator, measure_inference_time

__all__ = ['ModelEvaluator', 'measure_inference_time']

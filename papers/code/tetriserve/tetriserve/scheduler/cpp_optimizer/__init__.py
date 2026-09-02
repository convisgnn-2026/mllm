"""
C++ optimized scheduler components for FluidServe.
"""

try:
    from . import cpp_optimizer
except ImportError:
    cpp_optimizer = None

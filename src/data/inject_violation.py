"""Violation injection is implemented by :mod:`src.data.render_page`.

Keeping this module makes the pipeline boundary explicit while avoiding a second,
divergent renderer for the portfolio-sized build.
"""

from src.data.render_page import VIOLATIONS, render_one

__all__ = ["VIOLATIONS", "render_one"]


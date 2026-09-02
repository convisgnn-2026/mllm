"""
Configuration module for experiment plotting.
"""

from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import font_manager as fm


def validate_arial_font():
    """
    Validate that Arial font is available and register all Arial variants.
    Raises RuntimeError if Arial is not found.
    """
    # Register Arial Regular
    arial_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.TTF",
        "/usr/share/fonts/truetype/msttcorefonts/arial.TTF",
    ]
    arial_path = next((p for p in arial_candidates if Path(p).exists()), None)
    if not arial_path:
        raise RuntimeError(
            "Arial font file not found. Ensure msttcorefonts (or Arial TTF) is installed "
            "and available at a known path, then rerun this script."
        )

    fm.fontManager.addfont(arial_path)
    print(f"Using Arial Regular at: {arial_path}")

    # Register Arial Bold (essential for bold tick labels!)
    arial_bold_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/ARIALBD.TTF",
    ]
    arial_bold_path = next((p for p in arial_bold_candidates if Path(p).exists()), None)
    if arial_bold_path:
        fm.fontManager.addfont(arial_bold_path)
        print(f"Using Arial Bold at: {arial_bold_path}")
    else:
        print("WARNING: Arial Bold not found. Bold text may not render properly.")

    # Register Arial Italic (optional but good to have)
    arial_italic_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/ariali.ttf",
    ]
    arial_italic_path = next((p for p in arial_italic_candidates if Path(p).exists()), None)
    if arial_italic_path:
        fm.fontManager.addfont(arial_italic_path)
        print(f"Using Arial Italic at: {arial_italic_path}")

    # Register Arial Bold Italic (optional)
    arial_bold_italic_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf",
    ]
    arial_bold_italic_path = next((p for p in arial_bold_italic_candidates if Path(p).exists()), None)
    if arial_bold_italic_path:
        fm.fontManager.addfont(arial_bold_italic_path)
        print(f"Using Arial Bold Italic at: {arial_bold_italic_path}")


# Validate Arial font on module import
validate_arial_font()


class PlotConfig:
    """Configuration for plot appearance and behavior."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        show_reference_line: bool = True,
        reference_value: float = 0.95,
        dpi: int = 300,
        color_scheme: str = "academic",
        use_academic_style: bool = True,
    ):
        self.figsize = figsize
        self.show_reference_line = show_reference_line
        self.reference_value = reference_value
        self.dpi = dpi
        self.color_scheme = color_scheme
        self.use_academic_style = use_academic_style

        # Style settings - use white background for academic papers
        if use_academic_style:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams["font.family"] = "sans-serif"
            # plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
            plt.rcParams["font.sans-serif"] = ["Arial"]
        else:
            plt.style.use("seaborn-v0_8-darkgrid")


# Global configuration constants
PREFERRED_SCHEDULE_ORDER = [
    "dyn_slo_schedule_global_dp",
    "data_parallel",
    "fixed_sp_2",
    "fixed_sp_4",
    "fixed_sp_8",
    # "per_resolution_schedule",
    "per_resolution_schedule_1_1_2_8",
    "per_resolution_schedule_1_1_2_4",
    # "per_resolution_schedule_1_1_4_8",
]

SCHEDULE_NAME_MAPPING = {
    "data_parallel": "xDiT(SP=1)",
    "fixed_sp_2": "xDiT(SP=2)",
    "fixed_sp_4": "xDiT(SP=4)",
    "fixed_sp_8": "xDiT(SP=8)",
    "per_resolution_schedule": "RSSP",
    "dyn_slo_schedule_global_scale_up": "TetriServe (ours) + Scale-up",
    "dyn_slo_schedule_global": "TetriServe (ours)",
    "dyn_slo_schedule_global_dp": "TetriServe (ours) + DP",
}

# Plot styling constants
PLOT_MARKERS = {"slo": "o", "goodput": "s", "latency_mean": "^", "latency_p90": "D"}

PLOT_PARAMS = {
    "markersize": 8,
    "linewidth": 2,
    "alpha": 0.8,
    "grid_alpha": 0.3,
}


def remap_schedule_logic_name(schedule_logic: str) -> str:
    """
    Remap schedule logic names for better display in graphs.
    If no mapping exists, returns the original name.
    """
    return SCHEDULE_NAME_MAPPING.get(schedule_logic, schedule_logic)


def get_ordered_schedule_logics(available_logics: List[str]) -> List[str]:
    """
    Return schedule logics in preferred display order.
    TetriServe first, then xDiT variants in order, then any additional schedules.
    """
    available_set = set(available_logics)
    ordered_logics = []

    # Add schedules in preferred order if they exist
    for logic in PREFERRED_SCHEDULE_ORDER:
        if logic in available_set:
            ordered_logics.append(logic)
            available_set.remove(logic)
        else:
            print(f"Schedule logic {logic} not found in available logics, just skipping")
            continue

    # # Add any remaining schedules not in preferred order (sorted for consistency)
    # ordered_logics.extend(sorted(available_set))

    return ordered_logics


def get_colors_for_schedules(schedule_logics: List[str], color_scheme: str = "academic") -> np.ndarray:
    """Get colors for the given schedule logics."""
    if color_scheme == "academic":
        # Academic paper style colors: orange, light blue, grays
        academic_colors = ["#FF8C00", "#87CEEB", "#A9A9A9", "#696969", "#2F4F4F", "#808080"]
        # Extend with more grays if needed
        while len(academic_colors) < len(schedule_logics):
            academic_colors.append("#708090")
        return np.array([academic_colors[i] for i in range(len(schedule_logics))])
    else:
        # Use matplotlib colormap
        cmap = plt.cm.get_cmap(color_scheme)
        return cmap(np.linspace(0, 1, len(schedule_logics)))

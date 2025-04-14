import sys
import os

def get_resource_path(relative_path: str) -> str:
    """ Get the absolute path to a resource, works for dev and for PyInstaller.

    Args:
        relative_path: The relative path from the script's location or bundle root.

    Returns:
        The absolute path to the resource.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in a PyInstaller bundle, assume dev environment.
        # Use the directory containing this utils.py script (installer dir) as the base.
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

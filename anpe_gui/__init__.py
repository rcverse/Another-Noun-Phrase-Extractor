"""
ANPE GUI Package
"""

# Import version from dedicated module
from .version import __version__

# Make core components available at package level
from .app import main
from .main_window import MainWindow
from .splash_screen import SplashScreen 
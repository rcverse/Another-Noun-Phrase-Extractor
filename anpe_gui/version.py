"""
Version information for the ANPE GUI package.

This module is separated to avoid circular import issues.
It should not import any other modules from the package.
"""

# Version of the ANPE GUI package
__version__ = "0.1.0"

# Function to get version information
def get_version():
    """Return the version string."""
    return __version__ 
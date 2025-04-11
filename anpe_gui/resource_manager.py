"""
Resource management utilities for ANPE GUI.

This module provides a centralized way to access resources, both when running from source
and when running from a packaged application.
"""

from pathlib import Path
import os
from PyQt6.QtCore import QFile, QResource, QDir
from PyQt6.QtGui import QIcon, QPixmap

class ResourceManager:
    """
    Manages access to application resources with support for both development and packaged environments.
    """
    # Flag to determine if resources are embedded (will be set when resources are initialized)
    _using_embedded_resources = False

    @classmethod
    def initialize(cls):
        """
        Initialize the resource system. Assume resources are embedded via imported module,
        but keep fallback to file system access if needed.
        """
        # Check if resources are accessible via the Qt resource system (qrc:/ prefix).
        # We test this by trying to check for the existence of a known resource file.
        # QFile.exists() works with the qrc:/ paths.
        test_resource = QFile(":/icons/app_icon.png") # Use a known resource from your qrc
        if test_resource.exists():
            cls._using_embedded_resources = True
        else:
            # If the embedded resource check fails, assume we need the filesystem fallback.
            # This might happen during development if resources_rc.py hasn't been generated/imported
            # or if there's an issue with the resource system itself.
            cls._using_embedded_resources = False
            print("INFO: Embedded resources not found or accessible. Falling back to file system paths.", flush=True) # Optional: Add logging/print

    @classmethod
    def get_icon(cls, icon_name):
        """
        Get a QIcon for the specified resource.
        
        Args:
            icon_name: Name of the icon file (e.g., 'app_icon.png')
            
        Returns:
            QIcon: The loaded icon
        """
        if cls._using_embedded_resources:
            return QIcon(f":/icons/{icon_name}")
        else:
            # Fallback to file system
            icon_path = cls.get_resource_path(icon_name)
            return QIcon(str(icon_path)) # Ensure path is string

    @classmethod
    def get_pixmap(cls, image_name):
        """
        Get a QPixmap for the specified resource.
        
        Args:
            image_name: Name of the image file (e.g., 'app_icon.png')
            
        Returns:
            QPixmap: The loaded image
        """
        if cls._using_embedded_resources:
            return QPixmap(f":/icons/{image_name}")
        else:
            # Fallback to file system
            image_path = cls.get_resource_path(image_name)
            return QPixmap(str(image_path)) # Ensure path is string

    @classmethod
    def get_resource_path(cls, resource_name):
        """
        Get the full path to a resource file (for fallback).
        
        Args:
            resource_name: Name of the resource file (e.g., 'app_icon.png')
            
        Returns:
            Path: Path object to the resource
        """
        # When in development or fallback, resources are in the resources directory relative to this file
        resources_dir = Path(__file__).parent / "resources"
        return resources_dir / resource_name

    @classmethod
    def get_style_url(cls, resource_name):
        """
        Get the URL for a resource to be used in stylesheets.
        
        Args:
            resource_name: Name of the resource file (e.g., 'expand_open.svg')
            
        Returns:
            str: URL for the resource that works in Qt stylesheets
        """
        if cls._using_embedded_resources:
            # Use the simpler :/ prefix, which is often more reliable in stylesheets
            return f":/icons/{resource_name}"
        else:
            # For development/fallback mode, use relative path from the perspective
            # of where the application is run or how stylesheets resolve paths.
            # This might need adjustment depending on the context where it's used.
            # Using a relative path from the package root might be more robust.
            # Let's assume the style is applied from code within anpe_gui.
            # We need to ensure the path is relative to the CWD or correctly resolvable.
            # A simple relative path from the package root is often best.
            resource_path = cls.get_resource_path(resource_name)
            # Try to make it relative to the CWD if possible, otherwise use absolute
            try:
                relative_path = resource_path.relative_to(Path.cwd())
                return str(relative_path).replace("\\\\", "/") # Use forward slashes for URLs
            except ValueError:
                 # If not relative to CWD, use absolute path (might not always work in stylesheets)
                 # Using Path.as_uri() is safer for file URLs
                 return resource_path.as_uri() 
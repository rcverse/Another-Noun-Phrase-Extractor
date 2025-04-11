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
        Initialize the resource system. Check if Qt resource system is available,
        otherwise fall back to file system access.
        """
        # Check if resources are embedded in the PyQt resource system
        if QResource.registerResource("anpe_gui/resources.rcc"):
            cls._using_embedded_resources = True
        else:
            # Try alternate location for packaged app
            alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources.rcc")
            if os.path.exists(alt_path) and QResource.registerResource(alt_path):
                cls._using_embedded_resources = True
            else:
                cls._using_embedded_resources = False

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
            return QIcon(icon_path)

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
            return QPixmap(image_path)

    @classmethod
    def get_resource_path(cls, resource_name):
        """
        Get the full path to a resource file.
        
        Args:
            resource_name: Name of the resource file (e.g., 'app_icon.png')
            
        Returns:
            str: Full path to the resource
        """
        # When in development, resources are in the resources directory
        resources_dir = Path(__file__).parent / "resources"
        return str(resources_dir / resource_name)

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
            return f"qrc:/icons/{resource_name}"
        else:
            # For development mode, use relative path
            # Need to handle the path differently for stylesheets
            return f"anpe_gui/resources/{resource_name}" 
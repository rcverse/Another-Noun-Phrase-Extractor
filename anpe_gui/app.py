#!/usr/bin/env python3
"""
Main entry point for the ANPE GUI application.
"""

import sys
import os
from pathlib import Path
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer # Original import
from PyQt6.QtGui import QFont # Original import
from anpe_gui.main_window import MainWindow
from anpe_gui.splash_screen import SplashScreen
from anpe_gui.theme import apply_theme
from anpe_gui.resource_manager import ResourceManager
from anpe.utils.setup_models import check_all_models_present # Assuming this function exists
# from anpe_gui.setup_wizard import SetupWizard # REMOVED: Assuming this class will be created

# Import resource module
import anpe_gui.resources_rc

def main():
    """Launch the main application."""
    # Configure High-DPI scaling via environment variables
    # These must be set before QApplication is created
    # Note: In PyQt6, high-DPI scaling is enabled by default
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # QT_SCALE_FACTOR can be used to manually set scaling if needed 
    # os.environ["QT_SCALE_FACTOR"] = "1.5" # Only enable if needed
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("ANPE GUI")
    app.setOrganizationName("ANPE")
    
    # Initialize resource manager
    ResourceManager.initialize()
    
    # Adjust default font slightly based on screen DPI
    screen = app.primaryScreen()
    if screen:
        dpi = screen.logicalDotsPerInch()
        default_font = app.font()
        
        # On high-DPI screens, adjust the font size to be more readable
        if dpi > 120:  # Regular screens are typically 96-108 DPI; 4K screens are usually 150+ DPI
            dpi_scale_factor = min(dpi / 96.0, 1.5)  # Don't scale larger than 1.5x
            font_size_pt = max(9, round(9 * dpi_scale_factor))
            default_font.setPointSize(font_size_pt)
            app.setFont(default_font)
            print(f"Adjusted font size for high-DPI display: {font_size_pt}pt (DPI: {dpi})")
    
    apply_theme(app)

    main_window = None
    # setup_wizard = None # REMOVED: Keep track of the wizard instance

    # --- SplashScreen Setup ---
    splash = SplashScreen()
    # --- Start Splash Animation Immediately ---
    splash.start_loading_animation(app)
    print("APP: Splash animation initiated immediately after creation.")

    # --- Function to launch the main window ---
    def launch_main_window():
        nonlocal main_window
        if main_window is None:
            print("APP: Creating MainWindow instance...")
            main_window = MainWindow()
            print("APP: MainWindow instance created.")
        if main_window:
            main_window.fade_in() 

    # --- Function called after splash screen (or directly if setup needed) ---
    def proceed_after_splash_or_setup():
        nonlocal main_window # Reference the main_window instance

        print("APP: Checking if models are present...")
        models_present = check_all_models_present() # Check for models
        print(f"APP: Models present: {models_present}")

        if models_present:
            print("APP: Models found. Preparing to launch main window.")
            if splash: 
                # Connect splash fade-out completion to launching main window
                # Ensure we connect only once if this function could be called multiple times
                try: 
                    splash.fade_out_complete.disconnect(launch_main_window)
                except TypeError:
                    pass # No connection existed
                splash.fade_out_complete.connect(launch_main_window)
                # The splash screen will call fade_out() itself when its loading is done
                # We no longer close it here or launch the window directly
                print("APP: Waiting for splash fade out...")
            else: # If no splash was shown (e.g., during development/testing)
                launch_main_window()
        else:
            print("APP: Models not found. Launching Setup Wizard.")
            # --- MODIFIED: Instead of wizard, show message and launch main window in disabled state ---
            print("APP: Models not found. Will show message and launch main window (partially disabled).")
            # Ensure splash is closed before showing message/main window
            def show_message_and_main_window():
                # Message box is now handled inside MainWindow's init error handler
                # QMessageBox.warning(None, "Models Missing", 
                #                     "Required ANPE models (spaCy, Benepar, NLTK) are missing.\n" 
                #                     "Please use 'Manage Models' (gear icon) in the main window to install them.")
                # Launch the main window directly, it will handle its own state
                launch_main_window() 
            
            if splash:
                try:
                    splash.fade_out_complete.disconnect() # Disconnect previous connections
                except TypeError:
                    pass
                splash.fade_out_complete.connect(show_message_and_main_window)
                print("APP: Waiting for splash fade out before showing message/main window...")
                # Splash fade_out is triggered internally
            else:
                # No splash, show message and launch main window directly
                show_message_and_main_window()
            # -------------------------------------------------------------------------------------

            # --- OLD WIZARD LOGIC (REMOVED) --- 
            # if splash:
            #     # Ensure splash is closed *before* showing wizard. 
            #     # Since fade_out handles close, connect fade_out_complete to showing wizard.
            #     # We need a small helper function or lambda to show the wizard
            #     def show_wizard_after_splash():
            #         nonlocal setup_wizard
            #         if setup_wizard is None: # Create wizard only once
            #             setup_wizard = SetupWizard()
            #             setup_wizard.setup_complete.connect(launch_main_window) # Wizard success launches main window (with fade-in)
            #             setup_wizard.setup_cancelled.connect(app.quit)
            #         setup_wizard.show()
            #     
            #     try:
            #         splash.fade_out_complete.disconnect() # Disconnect any previous connections
            #     except TypeError:
            #         pass
            #     splash.fade_out_complete.connect(show_wizard_after_splash)
            #     # Splash fade_out is triggered internally when loading finishes
            #     print("APP: Waiting for splash fade out before showing wizard...")
            # 
            # else: # No splash, show wizard directly
            #     if setup_wizard is None: 
            #         setup_wizard = SetupWizard()
            #         setup_wizard.setup_complete.connect(launch_main_window)
            #         setup_wizard.setup_cancelled.connect(app.quit)
            #     setup_wizard.show()

    # --- Connect Splash Signal ---
    # Splash loading finished triggers the check/proceed logic
    splash.loading_finished.connect(proceed_after_splash_or_setup)

    # --- Start Splash (Moved earlier) ---
    # splash.start_loading_animation(app) # No longer needed here
    # print("APP: Splash animation started (includes fade-in).") # Commented out old message

    # --- Start Event Loop ---
    exit_code = app.exec()
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 
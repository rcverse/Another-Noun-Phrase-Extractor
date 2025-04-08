#!/usr/bin/env python3
"""
Main entry point for the ANPE GUI application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from anpe_gui.main_window import MainWindow
from anpe_gui.splash_screen import SplashScreen
from anpe_gui.theme import apply_theme
from anpe.utils.setup_models import check_all_models_present # Assuming this function exists
from anpe_gui.setup_wizard import SetupWizard # Assuming this class will be created

def main():
    """Launch the main application."""
    app = QApplication(sys.argv)
    app.setApplicationName("ANPE GUI")
    app.setOrganizationName("ANPE")
    
    # apply_theme(app) # Keep styles disabled for now during debugging

    main_window = None
    setup_wizard = None # Keep track of the wizard instance

    # --- SplashScreen Setup ---
    splash = SplashScreen()

    # --- Function to launch the main window ---
    def launch_main_window():
        nonlocal main_window
        if main_window is None:
            print("APP: Creating MainWindow instance...")
            main_window = MainWindow()
            print("APP: MainWindow instance created.")
        if main_window:
            main_window.show()

    # --- Function called after splash screen (or directly if setup needed) ---
    def proceed_after_splash_or_setup():
        nonlocal setup_wizard # Reference the wizard instance

        print("APP: Checking if models are present...")
        models_present = check_all_models_present() # Check for models
        print(f"APP: Models present: {models_present}")

        if models_present:
            print("APP: Models found. Launching main window.")
            launch_main_window()
            if splash: # Close splash if it was shown
                splash.close()
        else:
            print("APP: Models not found. Launching Setup Wizard.")
            if splash: # Close splash before showing wizard
                splash.close()
            if setup_wizard is None: # Create wizard only once
                setup_wizard = SetupWizard()
                # Connect the wizard's success signal to launching the main window
                setup_wizard.setup_complete.connect(launch_main_window)
                # Optional: Connect a signal for wizard cancellation/failure to exit app
                setup_wizard.setup_cancelled.connect(app.quit)
            setup_wizard.show()

    # --- Connect Splash Signal ---
    # The splash screen finishes loading, then we check models and proceed
    splash.loading_finished.connect(proceed_after_splash_or_setup)

    # --- Start Splash ---
    splash.start_loading_animation(app) # Show splash screen

    # --- Start Event Loop ---
    exit_code = app.exec()
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 
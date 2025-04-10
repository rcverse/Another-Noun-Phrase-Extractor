#!/usr/bin/env python3
"""
Main entry point for the ANPE GUI application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer # Import QTimer
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
    
    apply_theme(app)

    main_window = None
    setup_wizard = None # Keep track of the wizard instance

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
        nonlocal setup_wizard # Reference the wizard instance

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
            if splash:
                # Ensure splash is closed *before* showing wizard. 
                # Since fade_out handles close, connect fade_out_complete to showing wizard.
                # We need a small helper function or lambda to show the wizard
                def show_wizard_after_splash():
                    nonlocal setup_wizard
                    if setup_wizard is None: # Create wizard only once
                        setup_wizard = SetupWizard()
                        setup_wizard.setup_complete.connect(launch_main_window) # Wizard success launches main window (with fade-in)
                        setup_wizard.setup_cancelled.connect(app.quit)
                    setup_wizard.show()
                
                try:
                    splash.fade_out_complete.disconnect() # Disconnect any previous connections
                except TypeError:
                    pass
                splash.fade_out_complete.connect(show_wizard_after_splash)
                # Splash fade_out is triggered internally when loading finishes
                print("APP: Waiting for splash fade out before showing wizard...")

            else: # No splash, show wizard directly
                if setup_wizard is None: 
                    setup_wizard = SetupWizard()
                    setup_wizard.setup_complete.connect(launch_main_window)
                    setup_wizard.setup_cancelled.connect(app.quit)
                setup_wizard.show()

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
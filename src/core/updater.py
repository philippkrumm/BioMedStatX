"""
Auto-updater for BioMedStatX using GitHub Releases API
"""
import requests
import sys
from packaging import version
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# Current version - update this with each release
CURRENT_VERSION = "2.0" 
GITHUB_REPO = "philippkrumm/BioMedStatX"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

class UpdateChecker(QThread):
    """Thread to check for updates without blocking the UI"""
    update_available = pyqtSignal(dict)  # Sends update info
    no_update = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, current_version=CURRENT_VERSION):
        super().__init__()
        self.current_version = current_version
        
    def run(self):
        """Check for updates in background thread"""
        try:
            # Get latest release info from GitHub
            response = requests.get(GITHUB_API_URL, timeout=10)
            response.raise_for_status()
            
            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')  # Remove 'v' prefix if present
            
            # Compare versions
            if version.parse(latest_version) > version.parse(self.current_version):
                # Update available
                update_info = {
                    'version': latest_version,
                    'release_notes': release_data.get('body', ''),
                    'release_url': release_data.get('html_url', f'https://github.com/{GITHUB_REPO}/releases/latest'),
                    'release_data': release_data
                }
                
                self.update_available.emit(update_info)
            else:
                self.no_update.emit()
                
        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"Network error checking for updates: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Error checking for updates: {str(e)}")

class AutoUpdater:
    """Main updater class"""
    
    def __init__(self, parent_widget=None):
        self.parent = parent_widget
        self.update_checker = None
        
    def check_for_updates(self, silent=False):
        """Check for updates - silent=True suppresses 'no update' messages"""
        if self.update_checker and self.update_checker.isRunning():
            return
            
        self.silent = silent
        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self._on_update_available)
        self.update_checker.no_update.connect(self._on_no_update)
        self.update_checker.error_occurred.connect(self._on_error)
        self.update_checker.start()
        
    def _on_update_available(self, update_info):
        """Handle when update is available"""
        version_str = update_info['version']
        release_notes = update_info['release_notes']
        release_url = update_info['release_url']
        
        # Show update notification with link to releases page
        msg = QMessageBox(self.parent)
        msg.setWindowTitle("Update Available")
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"A new version {version_str} is available!")
        
        release_preview = release_notes[:400] if release_notes else "No release notes available."
        if len(release_notes) > 400:
            release_preview += "..."
            
        msg.setInformativeText(
            f"Current version: {CURRENT_VERSION}\n"
            f"New version: {version_str}\n\n"
            f"Release Notes:\n{release_preview}\n\n"
            f"Please visit the GitHub releases page to download the update."
        )
        
        # Add button to open releases page
        msg.addButton("Open Releases Page", QMessageBox.AcceptRole)
        msg.addButton("Later", QMessageBox.RejectRole)
        
        if msg.exec_() == 0:  # Open Releases Page clicked
            import webbrowser
            webbrowser.open(release_url)
    
    def _on_no_update(self):
        """Handle when no update is available"""
        if not self.silent:
            QMessageBox.information(
                self.parent,
                "No Updates",
                f"You are running the latest version ({CURRENT_VERSION})."
            )
    
    def _on_error(self, error_message):
        """Handle update check errors"""
        if not self.silent:
            QMessageBox.warning(
                self.parent,
                "Update Check Failed",
                f"Could not check for updates:\n{error_message}"
            )

def add_update_menu_to_app(main_window):
    """Add update functionality to existing menu"""
    updater = AutoUpdater(main_window)
    
    # Add to Help menu
    help_menu = None
    for action in main_window.menuBar().actions():
        if action.text() == "Help":
            help_menu = action.menu()
            break
    
    if help_menu:
        help_menu.addSeparator()
        
        # Check for updates action
        check_updates_action = help_menu.addAction("Check for Updates...")
        check_updates_action.triggered.connect(lambda: updater.check_for_updates(silent=False))
        
        # Auto-check on startup (after 5 seconds delay)
        startup_timer = QTimer()
        startup_timer.singleShot(5000, lambda: updater.check_for_updates(silent=True))
    
    return updater
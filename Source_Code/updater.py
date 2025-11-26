"""
Auto-updater for BioMedStatX using GitHub Releases API
"""
import requests
import json
import os
import sys
import subprocess
import tempfile
import zipfile
from packaging import version
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import logging

# Current version - update this with each release
CURRENT_VERSION = "1.0.1" 
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
                    'download_url': None,
                    'release_data': release_data
                }
                
                # Find suitable download asset (Windows .exe, Mac .dmg, etc.)
                for asset in release_data.get('assets', []):
                    asset_name = asset['name'].lower()
                    if sys.platform == "win32" and asset_name.endswith('.exe'):
                        update_info['download_url'] = asset['browser_download_url']
                        break
                    elif sys.platform == "darwin" and (asset_name.endswith('.dmg') or asset_name.endswith('.pkg')):
                        update_info['download_url'] = asset['browser_download_url']
                        break
                    elif sys.platform.startswith("linux") and (asset_name.endswith('.AppImage') or asset_name.endswith('.tar.gz')):
                        update_info['download_url'] = asset['browser_download_url']
                        break
                
                self.update_available.emit(update_info)
            else:
                self.no_update.emit()
                
        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"Network error checking for updates: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Error checking for updates: {str(e)}")

class UpdateDownloader(QThread):
    """Thread to download and install updates"""
    progress_updated = pyqtSignal(int)  # Progress percentage
    download_complete = pyqtSignal(str)  # Downloaded file path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, download_url, filename):
        super().__init__()
        self.download_url = download_url
        self.filename = filename
        
    def run(self):
        """Download the update file"""
        try:
            response = requests.get(self.download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Create temp file
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, self.filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            self.progress_updated.emit(progress)
            
            self.download_complete.emit(file_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Error downloading update: {str(e)}")

class AutoUpdater:
    """Main updater class"""
    
    def __init__(self, parent_widget=None):
        self.parent = parent_widget
        self.update_checker = None
        self.update_downloader = None
        
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
        download_url = update_info['download_url']
        
        if not download_url:
            QMessageBox.information(
                self.parent,
                "Update Available",
                f"A new version {version_str} is available!\n\n"
                f"Please visit the GitHub releases page to download it manually.\n\n"
                f"Release Notes:\n{release_notes[:500]}..."
            )
            return
        
        # Show update dialog
        msg = QMessageBox(self.parent)
        msg.setWindowTitle("Update Available")
        msg.setText(f"A new version {version_str} is available!")
        msg.setInformativeText(f"Current version: {CURRENT_VERSION}\n\n"
                              f"Would you like to download and install the update?\n\n"
                              f"Release Notes:\n{release_notes[:300]}...")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        
        if msg.exec_() == QMessageBox.Yes:
            self._download_update(download_url, update_info)
    
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
    
    def _download_update(self, download_url, update_info):
        """Download and install update"""
        filename = os.path.basename(download_url)
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog(
            "Downloading update...", "Cancel", 0, 100, self.parent
        )
        self.progress_dialog.setWindowTitle("Downloading Update")
        self.progress_dialog.show()
        
        # Start download
        self.update_downloader = UpdateDownloader(download_url, filename)
        self.update_downloader.progress_updated.connect(self.progress_dialog.setValue)
        self.update_downloader.download_complete.connect(self._on_download_complete)
        self.update_downloader.error_occurred.connect(self._on_download_error)
        self.progress_dialog.canceled.connect(self.update_downloader.terminate)
        self.update_downloader.start()
    
    def _on_download_complete(self, file_path):
        """Handle completed download"""
        self.progress_dialog.close()
        
        msg = QMessageBox(self.parent)
        msg.setWindowTitle("Download Complete")
        msg.setText("Update downloaded successfully!")
        msg.setInformativeText(f"The update has been saved to:\n{file_path}\n\n"
                              "Would you like to install it now?\n"
                              "(This will close the current application)")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        if msg.exec_() == QMessageBox.Yes:
            self._install_update(file_path)
    
    def _on_download_error(self, error_message):
        """Handle download errors"""
        self.progress_dialog.close()
        QMessageBox.critical(
            self.parent,
            "Download Failed",
            f"Failed to download update:\n{error_message}"
        )
    
    def _install_update(self, file_path):
        """Install the downloaded update"""
        try:
            if sys.platform == "win32":
                # Windows: Run installer
                subprocess.Popen([file_path])
            elif sys.platform == "darwin":
                # macOS: Open installer
                subprocess.Popen(["open", file_path])
            else:
                # Linux: Make executable and run
                os.chmod(file_path, 0o755)
                subprocess.Popen([file_path])
            
            # Close current application
            QApplication.quit()
            
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Installation Failed",
                f"Could not start installer:\n{str(e)}\n\n"
                f"Please run the installer manually:\n{file_path}"
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
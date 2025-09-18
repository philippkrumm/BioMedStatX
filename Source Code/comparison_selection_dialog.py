from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QDialogButtonBox, QWidget, QScrollArea, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

class ComparisonSelectionDialog(QDialog):
    def __init__(self, comparisons, parent=None, checked_by_default=True):
        super().__init__(parent)
        self.setWindowTitle("Select Group Comparisons for Post-hoc Test")
        self.selected = set()
        self.comparisons = comparisons
        self.checkboxes = []
        self.checked_by_default = checked_by_default
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Select the group comparisons you want to include in the post-hoc test:")
        layout.addWidget(label)

        # Scroll area for many comparisons
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for comp in self.comparisons:
            text = f"{comp[0]}  vs  {comp[1]}"
            cb = QCheckBox(text)
            cb.setChecked(self.checked_by_default)
            self.checkboxes.append(cb)
            scroll_layout.addWidget(cb)
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        
        # Limit maximum height to prevent dialog from becoming too large
        # Calculate dynamic height: max 400px or 60% of screen height, whichever is smaller
        from PyQt5.QtWidgets import QApplication
        if QApplication.instance():
            screen = QApplication.instance().primaryScreen()
            if screen:
                screen_height = screen.geometry().height()
                max_height = min(400, int(screen_height * 0.6))
                scroll.setMaximumHeight(max_height)
            else:
                scroll.setMaximumHeight(400)  # Fallback
        else:
            scroll.setMaximumHeight(400)  # Fallback
            
        layout.addWidget(scroll)

        # Select/Deselect All buttons
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self):
        """Select all checkboxes"""
        for cb in self.checkboxes:
            cb.setChecked(True)
    
    def _deselect_all(self):
        """Deselect all checkboxes"""
        for cb in self.checkboxes:
            cb.setChecked(False)

    def get_selected_comparisons(self):
        selected = []
        for cb, comp in zip(self.checkboxes, self.comparisons):
            if cb.isChecked():
                selected.append(comp)
        return selected

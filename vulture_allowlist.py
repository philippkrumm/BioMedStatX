# vulture_allowlist.py
# ====================
# Funktionen/Attribute die vulture fälschlicherweise als "unused" meldet,
# weil sie dynamisch verwendet werden (Qt-Slots, getattr, Plugins, etc.).
#
# Format: Jeder Eintrag täuscht vulture durch eine scheinbare Verwendung.
# Neue false-positives hier eintragen statt --min-confidence zu senken.

# --- PyQt / Qt Slots & Event-Handler ---
# Qt ruft diese per Signal-Slot-Mechanismus auf — nie direkt im Code sichtbar.
_.closeEvent          # noqa
_.resizeEvent         # noqa
_.paintEvent          # noqa
_.mousePressEvent     # noqa
_.keyPressEvent       # noqa
_.showEvent           # noqa
_.hideEvent           # noqa

# --- Matplotlib Callbacks ---
_.on_press            # noqa
_.on_release          # noqa

# --- Dynamisch genutzte Attribute (getattr / **kwargs) ---
# Falls vulture eine Methode meldet die via getattr() aufgerufen wird,
# hier eintragen:
# _.method_name       # noqa

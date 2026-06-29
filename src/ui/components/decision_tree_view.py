import math
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QColor, QPen, QBrush, QFont, QFontMetrics, QPolygonF, QPainter,
    QStaticText, QTextOption, QPalette
)
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QApplication
)

def get_line_intersection(x1, y1, x2, y2, w_src, h_src, w_tgt, h_tgt):
    """
    Calculates the exact exit point on the source node's boundary
    and the entrance point on the target node's boundary using vector math.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return x1, y1, x2, y2
        
    t_x_src = w_src / (2.0 * abs(dx)) if dx != 0 else float('inf')
    t_y_src = h_src / (2.0 * abs(dy)) if dy != 0 else float('inf')
    t_src = min(t_x_src, t_y_src)
    t_src = max(0.0, min(1.0, t_src))
    
    ex1 = x1 + t_src * dx
    ey1 = y1 + t_src * dy
    
    t_x_tgt = 1.0 - w_tgt / (2.0 * abs(dx)) if dx != 0 else float('-inf')
    t_y_tgt = 1.0 - h_tgt / (2.0 * abs(dy)) if dy != 0 else float('-inf')
    t_tgt = max(t_x_tgt, t_y_tgt)
    t_tgt = max(0.0, min(1.0, t_tgt))
    
    ex2 = x1 + t_tgt * dx
    ey2 = y1 + t_tgt * dy
    
    return ex1, ey1, ex2, ey2


def node_metrics(label, is_active):
    """Node box size + font, mirroring the HTML/SVG report renderer so the
    in-app tree matches the exported one. See report_single.html.j2."""
    lines = str(label or "").split('\n')
    fs = 15 if is_active else 12
    lh = fs * 1.3
    max_chars = max((len(line) for line in lines), default=0)
    est_text_w = max_chars * fs * 0.57
    min_w = 155.0 if is_active else 130.0
    min_h = 60.0 if is_active else 52.0
    node_w = max(min_w, min(500.0, est_text_w + 36.0))
    node_h = max(min_h, min(500.0, len(lines) * lh + 34.0))
    return node_w, node_h, fs


class DecisionNodeItem(QGraphicsItem):
    """
    Renders a dynamic decision node. Caches text layouts via QStaticText
    and activates DeviceCoordinateCache to maintain a high framerate during zoom/pan.
    """
    def __init__(self, node_id, label, x, y, is_active, is_square, is_dark, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.label = label
        self.is_active = is_active
        self.is_square = is_square
        self.is_dark = is_dark

        # Dimensions mirror the HTML/SVG renderer for a consistent look.
        self.w, self.h, self.fs = node_metrics(label, is_active)

        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.setToolTip(self.label.replace('\n', ' · '))

        # Cache item rendering as device coordinates
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self):
        return QRectF(-self.w / 2.0, -self.h / 2.0, self.w, self.h)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Color palette depending on active state and theme
        if self.is_dark:
            if self.is_active:
                fill_color = QColor("#133835")      # Muted Teal transparent
                border_color = QColor("#2dd4bf")    # Neon Teal accent
                text_color = QColor("#2dd4bf")
                border_width = 2.0
            else:
                fill_color = QColor("#162428")      # Dark Surface
                border_color = QColor("#243538")    # Fine boundary
                text_color = QColor("#8ba4ac")      # Muted text
                border_width = 1.0
        else:
            # Match the HTML/SVG report colors exactly (rgba over white).
            if self.is_active:
                fill_color = QColor(15, 118, 110, 31)    # rgba(15,118,110,0.12)
                border_color = QColor("#0f766e")         # Deep Teal accent
                text_color = QColor("#0f766e")
                border_width = 2.3
            else:
                fill_color = QColor(22, 49, 58, 8)        # rgba(22,49,58,0.03)
                border_color = QColor(22, 49, 58, 51)     # rgba(22,49,58,0.20)
                text_color = QColor(22, 49, 58, 173)      # rgba(22,49,58,0.68)
                border_width = 1.2

        # Draw background card
        pen = QPen(border_color, border_width)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        rx = 5.0 if self.is_square else 16.0
        
        # Adjust drawing rect by half pen width so the border doesn't get clipped
        adj = border_width / 2.0
        draw_rect = self.boundingRect().adjusted(adj, adj, -adj, -adj)
        painter.drawRoundedRect(draw_rect, rx, rx)

        # Draw multi-line text. drawText honours explicit '\n' line breaks
        # (QStaticText collapsed them, joining words like "participantsin").
        font = QFont("Segoe UI")
        font.setPixelSize(int(self.fs))
        if self.is_active:
            font.setBold(True)
        painter.setFont(font)
        painter.setPen(text_color)

        text_rect = self.boundingRect().adjusted(6.0, 4.0, -6.0, -4.0)
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, self.label)


class DecisionEdgeItem(QGraphicsItem):
    """
    Renders connection paths and exact rotated arrowheads.
    Alternative (non-taken) branch edges are drawn as dashed lines.
    """
    def __init__(self, source_node, target_node, is_active, is_dark, is_alternative=False, parent=None):
        super().__init__(parent)
        self.source_node = source_node
        self.target_node = target_node
        self.is_active = is_active
        self.is_dark = is_dark
        self.is_alternative = is_alternative
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self):
        p1 = self.source_node.pos()
        p2 = self.target_node.pos()
        min_x = min(p1.x(), p2.x()) - 20.0
        max_x = max(p1.x(), p2.x()) + 20.0
        min_y = min(p1.y(), p2.y()) - 20.0
        max_y = max(p1.y(), p2.y()) + 20.0
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        
        p1 = self.source_node.pos()
        p2 = self.target_node.pos()
        
        w_src, h_src = self.source_node.w, self.source_node.h
        w_tgt, h_tgt = self.target_node.w, self.target_node.h
        
        # Calculate intersection offset slightly outside node margins
        ex1, ey1, ex2, ey2 = get_line_intersection(
            p1.x(), p1.y(), p2.x(), p2.y(),
            w_src + 6.0, h_src + 6.0,
            w_tgt + 6.0, h_tgt + 6.0
        )
        
        dx = ex2 - ex1
        dy = ey2 - ey1
        if dx == 0 and dy == 0:
            return
            
        # Select palette colors
        if self.is_dark:
            if self.is_active:
                color = QColor("#2dd4bf")
            elif self.is_alternative:
                color = QColor("#3d6068")   # Visible but muted teal-grey
            else:
                color = QColor("#243538")
        else:
            # Mirror the HTML/SVG report edge colors exactly.
            if self.is_active:
                color = QColor("#0f766e")
            elif self.is_alternative:
                color = QColor(22, 49, 58, 82)    # rgba(22,49,58,0.32)
            else:
                color = QColor(22, 49, 58, 33)    # rgba(22,49,58,0.13)

        # Stroke widths mirror the HTML renderer.
        width = 2.8 if self.is_active else (1.5 if self.is_alternative else 1.2)
        pen = QPen(color, width)
        if self.is_alternative and not self.is_active:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)

        # Draw line path
        # Shorten line by arrow length to avoid bleed-through
        arrow_len = 9.0 * width
        L = math.hypot(dx, dy)
        if L > arrow_len:
            line_ex2 = ex2 - (dx / L) * arrow_len
            line_ey2 = ey2 - (dy / L) * arrow_len
            painter.drawLine(QPointF(ex1, ey1), QPointF(line_ex2, line_ey2))
        else:
            painter.drawLine(QPointF(ex1, ey1), QPointF(ex2, ey2))

        # Arrowhead. SVG markers scale with stroke width (markerUnits="strokeWidth"):
        # marker is 9×7 units, so the px size is 9*width long by 7*width tall —
        # that is why the HTML arrows look much larger than the old fixed 7px ones.
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        painter.save()
        painter.translate(ex2, ey2)
        painter.rotate(angle_deg)

        arrow_len = 9.0 * width
        half_base = 3.5 * width
        p_tip = QPointF(0, 0)
        p_bottom_left = QPointF(-arrow_len, -half_base)
        p_bottom_right = QPointF(-arrow_len, half_base)

        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([p_tip, p_bottom_left, p_bottom_right]))
        painter.restore()


class InteractiveDecisionTreeWidget(QGraphicsView):
    """
    Interactive widget for exploring statistical decision trees.
    Includes mouse wheel exponential scaling, drag-to-pan, and DPI-aware layout.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self._zoom_factor = 1.15
        self._min_zoom = 0.2
        self._max_zoom = 5.0
        
        self.tree_data = None
        self.user_interacted = False
        self.show_placeholder("Statistical decision path will appear here after the analysis.")

    def show_placeholder(self, text):
        self.scene.clear()
        self.tree_data = None
        self.user_interacted = False
        
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.Window).lightness() < 128
        bg_color = QColor("#0f1a1c") if is_dark else QColor("#ffffff")
        self.setBackgroundBrush(QBrush(bg_color))
        
        # Match the app-wide empty-state language (lblEmptyState): small, italic,
        # muted, centered — instead of a large left-aligned block that competes
        # with the panel subtitle above the canvas.
        placeholder_font = QFont("Segoe UI", 11)
        placeholder_font.setItalic(True)
        self.placeholder_item = self.scene.addText("", placeholder_font)
        self.placeholder_item.setTextWidth(360)
        option = QTextOption(Qt.AlignCenter)
        option.setWrapMode(QTextOption.WordWrap)
        self.placeholder_item.document().setDefaultTextOption(option)
        self.placeholder_item.setPlainText(text)
        self.placeholder_item.setDefaultTextColor(QColor("#8ba4ac") if is_dark else QColor("#8aacbf"))

        self.resetTransform()
        rect = self.placeholder_item.boundingRect()
        self.placeholder_item.setPos(0, 0)
        self.setSceneRect(0, 0, rect.width(), rect.height())
        self.setAlignment(Qt.AlignCenter)

    def set_tree_data(self, tree_json):
        self.scene.clear()
        self.tree_data = tree_json
        self.user_interacted = False
        # Restore default centering for the actual tree (placeholder uses AlignTop).
        self.setAlignment(Qt.AlignCenter)
        
        if not tree_json or "nodes" not in tree_json or not tree_json["nodes"]:
            self.show_placeholder("No decision tree data available.")
            return
            
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.Window).lightness() < 128
        bg_color = QColor("#0f1a1c") if is_dark else QColor("#ffffff")
        self.setBackgroundBrush(QBrush(bg_color))
        
        nodes = tree_json["nodes"]
        edges = tree_json["edges"]
        
        # Calculate dynamic bounds based on layout
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        SCALE_X = 64.0
        SCALE_Y = 64.0 * 1.7
        PAD = 96.0

        # Largest node half-extent shifts the whole layout clear of the edges,
        # exactly like the HTML renderer's maxNodeHalfW / maxNodeHalfH.
        max_half_w, max_half_h = 65.0, 28.0
        for node in nodes:
            nw, nh, _fs = node_metrics(node["label"], node["isActive"])
            max_half_w = max(max_half_w, nw / 2.0)
            max_half_h = max(max_half_h, nh / 2.0)

        def to_qt_coords(x, y):
            qx = (x - min_x) * SCALE_X + PAD + max_half_w
            qy = (max_y - y) * SCALE_Y + PAD + max_half_h
            return qx, qy
            
        node_map = {}
        active_items = []
        
        # Instantiating nodes first so edges can fetch node sizes dynamically
        for node in nodes:
            nid = node["id"]
            lbl = node["label"]
            is_active = node["isActive"]
            is_square = node["isSquare"]
            
            cx, cy = to_qt_coords(node["x"], node["y"])
            node_item = DecisionNodeItem(nid, lbl, cx, cy, is_active, is_square, is_dark)
            self.scene.addItem(node_item)
            node_map[nid] = node_item
            if is_active:
                active_items.append(node_item)
                
        # Instantiating edges
        for edge in edges:
            src_id = edge["source"]
            tgt_id = edge["target"]
            is_active = edge["isActive"]
            
            src_item = node_map.get(src_id)
            tgt_item = node_map.get(tgt_id)
            is_alternative = bool(edge.get("isAlternative", False))
            
            if src_item and tgt_item:
                edge_item = DecisionEdgeItem(src_item, tgt_item, is_active, is_dark, is_alternative)
                self.scene.addItem(edge_item)
                
        # Define scene bounding rectangle
        scene_rect = self.scene.itemsBoundingRect()
        margin_width = 60.0
        scene_rect.adjust(-margin_width, -margin_width, margin_width, margin_width)
        self.setSceneRect(scene_rect)
        
        # Zoom fit active segment path
        self.resetTransform()
        if active_items:
            rect = active_items[0].sceneBoundingRect()
            for item in active_items[1:]:
                rect = rect.united(item.sceneBoundingRect())
            rect.adjust(-60.0, -60.0, 60.0, 60.0)
            self.fitInView(rect, Qt.KeepAspectRatio)
        else:
            self.fitInView(scene_rect, Qt.KeepAspectRatio)

    def refit_view(self):
        if not self.tree_data:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            return
            
        active_rect = QRectF()
        has_active = False
        
        for item in self.scene.items():
            if isinstance(item, DecisionNodeItem) and item.is_active:
                if not has_active:
                    active_rect = item.sceneBoundingRect()
                    has_active = True
                else:
                    active_rect = active_rect.united(item.sceneBoundingRect())
                    
        if has_active:
            active_rect.adjust(-60.0, -60.0, 60.0, 60.0)
            self.fitInView(active_rect, Qt.KeepAspectRatio)
        else:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        self.user_interacted = True
        
        # Exponential zoom scaling direction
        if event.angleDelta().y() > 0:
            factor = self._zoom_factor
        else:
            factor = 1.0 / self._zoom_factor
            
        current_scale = self.transform().m11()
        if self._min_zoom <= current_scale * factor <= self._max_zoom:
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        self.user_interacted = True
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.tree_data and not self.user_interacted:
            self.refit_view()

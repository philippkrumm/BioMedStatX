import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QRectF
from ui.components.decision_tree_view import (
    InteractiveDecisionTreeWidget, DecisionNodeItem, DecisionEdgeItem
)

# A single QApplication instance must exist for the process
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_tree_data():
    return {
        "nodes": [
            {"id": "A", "x": 0.0, "y": 14.0, "label": "Start", "isActive": True, "isSquare": True},
            {"id": "B", "x": 0.0, "y": 12.5, "label": "Check Assumptions\nNormal: True\nEqual: True", "isActive": True, "isSquare": True},
            {"id": "C", "x": 0.0, "y": 11.0, "label": "Assumptions: Met", "isActive": True, "isSquare": True},
            {"id": "D1", "x": -2.0, "y": 9.5, "label": "No Transformation\nNeeded", "isActive": True, "isSquare": True},
            {"id": "D2", "x": 2.0, "y": 9.5, "label": "Apply Transformation", "isActive": False, "isSquare": True},
        ],
        "edges": [
            {"source": "A", "target": "B", "isActive": True},
            {"source": "B", "target": "C", "isActive": True},
            {"source": "C", "target": "D1", "isActive": True},
            {"source": "C", "target": "D2", "isActive": False},
        ]
    }


def test_placeholder_state(qapp):
    widget = InteractiveDecisionTreeWidget()
    assert widget.tree_data is None
    assert len(widget.scene.items()) > 0
    # The placeholder text item should be visible
    placeholder = widget.placeholder_item
    assert placeholder is not None
    assert placeholder.toPlainText().startswith("Statistical")


def test_tree_rendering_and_limits(qapp, mock_tree_data):
    widget = InteractiveDecisionTreeWidget()
    widget.set_tree_data(mock_tree_data)
    
    assert widget.tree_data == mock_tree_data
    scene = widget.scene
    
    # 5 nodes + 4 edges = 9 items
    items = scene.items()
    assert len(items) == 9
    
    # Verify bounds are valid
    scene_rect = scene.sceneRect()
    assert scene_rect.width() > 100
    assert scene_rect.height() > 100
    
    # All nodes must lie inside the scene boundaries
    for item in items:
        if isinstance(item, DecisionNodeItem):
            assert scene_rect.contains(item.sceneBoundingRect())


def test_node_collisions(qapp, mock_tree_data):
    widget = InteractiveDecisionTreeWidget()
    widget.set_tree_data(mock_tree_data)
    scene = widget.scene
    
    # Fetch all node items
    nodes = [item for item in scene.items() if isinstance(item, DecisionNodeItem)]
    assert len(nodes) == 5
    
    # No two nodes should collide/overlap
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i+1:]:
            # Check intersection of bounding rects
            rect_a = node_a.sceneBoundingRect()
            rect_b = node_b.sceneBoundingRect()
            assert not rect_a.intersects(rect_b), f"Collision detected between node {node_a.node_id} and {node_b.node_id}"


def test_memory_leak_and_cleanup(qapp, mock_tree_data):
    widget = InteractiveDecisionTreeWidget()
    scene = widget.scene
    
    # Render the tree repeatedly to verify cleanup works and no orphan items remain in the scene
    for _ in range(50):
        widget.set_tree_data(mock_tree_data)
        assert len(scene.items()) == 9
        
    # Clear view to placeholder and check count
    widget.show_placeholder("Cleared")
    assert len(scene.items()) == 1  # only placeholder text item


def test_association_decision_trees(qapp):
    from visualization.decisiontreevisualizer import DecisionTreeVisualizer
    
    # Test cases for each association model type
    models_to_test = ["Correlation", "LinearRegression", "ANCOVA", "LMM", "LogisticRegression", "CorrelationMatrix"]
    
    for model in models_to_test:
        results = {
            "model_type": model,
            "test": f"Test {model}",
            "p_value": 0.02,
            "alpha": 0.05,
            "r": 0.6,
            "r_squared": 0.45,
            "f_p_value": 0.01,
            "converged": True,
            "aic": 100,
            "bic": 110,
            "variables": ["X", "Y", "Z"]
        }
        
        tree_json = DecisionTreeVisualizer.get_tree_json(results)
        assert tree_json is not None, f"Failed to generate JSON tree for {model}"
        assert "nodes" in tree_json, f"Missing nodes in tree for {model}"
        assert "edges" in tree_json, f"Missing edges in tree for {model}"
        assert "tree_meta" in tree_json, f"Missing tree_meta in tree for {model}"
        assert "calculated_tier" in tree_json["tree_meta"]
        assert "total_samples" in tree_json["tree_meta"]
        assert len(tree_json["nodes"]) > 0, f"Empty nodes list for {model}"
        assert len(tree_json["edges"]) > 0, f"Empty edges list for {model}"
        
        # Verify node/edge schemas
        for node in tree_json["nodes"]:
            assert "id" in node
            assert "x" in node
            assert "y" in node
            assert "label" in node
            assert "isActive" in node
            assert "active" in node
            assert "isSquare" in node
            
        for edge in tree_json["edges"]:
            assert "source" in edge
            assert "from" in edge
            assert "target" in edge
            assert "to" in edge
            assert "isActive" in edge
            assert "active" in edge


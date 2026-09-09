from jinja2 import Environment, FileSystemLoader
import json
import math

env = Environment(loader=FileSystemLoader('src/templates'))
template = env.get_template('report_single.html.j2')
context = {
    "mode": "single",
    "report_title": "Test Report",
    "subtitle": "Test Subtitle",
    "hero": {"test_name": "Test", "p_value_display": "p<0.05", "effect_label": "d", "effect_size_display": "1.0", "significance_class": "is-significant", "significance_label": "Significant", "summary_note": "Summary"},
    "decision_tree_json": '{"nodes": [{"id": 1, "label": "test", "x": 0, "y": 0, "isActive": true}], "edges": []}',
    "bracket_data_json": "[]",
    "group_order_json": "[]",
    "decision_path": [{"active": True, "title": "Step 1"}]
}
html = template.render(context=context, mode="single")
print(html[:500])

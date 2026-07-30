import json
from export.html_exporter import HTMLExporter

results = {
    "test": "t-test (independent)",
    "model_type": "IndependentGroups",
    "statistic": 4.5,
    "mean_difference": 4.6,
    "confidence_interval": [4.2, 5.0],
    "p_value": 0.001,
    "effect_size": 1.9981,
    "effect_size_type": "cohen_d"
}

exporter = HTMLExporter()
rows = exporter._build_statistical_rows(results)
print(json.dumps(rows, indent=2))

"""
flowchartvisualizer.py
======================
Clean vertical flowchart visualizer for sequential / pipeline statistical
workflows:  LinearRegression, LogisticRegression, Correlation,
CorrelationMatrix, ANCOVA, and LMM.

Layout rules
------------
* Main trunk nodes sit on the vertical spine  x = 0.
* Mutually-exclusive alternative nodes are offset to x = ±1.3 so they stay
  tight to the trunk without overlap.
* Non-taken alternative branches carry  isAlternative = True  in the JSON
  output; both the Qt desktop widget (DecisionEdgeItem) and the HTML SVG
  canvas renderer render these as dashed lines.

Public API (matches DecisionTreeVisualizer's interface)
-------------------------------------------------------
FlowchartVisualizer.get_tree_json(results)  -> dict | None
    Returns {"nodes": [...], "edges": [...], "tree_meta": {...}}
    Compatible with InteractiveDecisionTreeWidget.set_tree_data() and the
    HTML canvas renderer (const TREE_DATA = ...).

FlowchartVisualizer.visualize(results, output_path=None) -> str | None
    Saves a matplotlib PNG and returns the file path.

FlowchartVisualizer.generate_and_save(results) -> str | None
    Thin wrapper that matches the DecisionTreeVisualizer API used by the
    HTML export path.
"""

from __future__ import annotations

import os
import tempfile

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yn(val: bool | None) -> str:
    if val is True:
        return "Yes"
    if val is False:
        return "No"
    return "n/a"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return ""
    return f"p = {p:.4f}"


def _fmt_r2(r2: float | None) -> str:
    if r2 is None:
        return ""
    return f"R² = {r2:.3f}"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FlowchartVisualizer:
    """
    Generates vertical flowchart visualisations for sequential statistical
    workflows (LinearRegression, LogisticRegression, Correlation,
    CorrelationMatrix, ANCOVA, LMM).

    The JSON schema emitted by get_tree_json() is identical to the one
    consumed by InteractiveDecisionTreeWidget.set_tree_data() and by the
    HTML canvas renderer, so no changes are required in those consumers.

    New JSON field on edge objects
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``isAlternative`` (bool) — True when the edge represents a branch that
    was *not* taken (e.g. "Standard OLS" when HC3 was selected).  Both the
    Qt and the HTML/SVG renderers draw these as dashed lines while keeping
    them visually distinct from completely inactive background edges.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_tree_json(results: dict) -> dict | None:
        """Return the JSON-serialisable tree dict for the interactive canvas.

        Node fields:   id, x, y, label, isActive, isSquare
        Edge fields:   source, target, isActive, isAlternative
        """
        try:
            model_type = results.get("model_type", "")
            alpha = float(results.get("alpha", 0.05))
            nodes_info, edges, highlighted, alternatives, tree_meta = (
                FlowchartVisualizer._build_topology(results, model_type, alpha)
            )

            active_nodes: set[str] = set()
            for u, v in highlighted:
                active_nodes.add(u)
                active_nodes.add(v)

            node_list = [
                {
                    "id":       nid,
                    "x":        float(info["pos"][0]),
                    "y":        float(info["pos"][1]),
                    "label":    info["label"],
                    "isActive": nid in active_nodes,
                    "isSquare": info["isSquare"],
                }
                for nid, info in nodes_info.items()
            ]

            edge_list = [
                {
                    "source":        u,
                    "target":        v,
                    "isActive":      (u, v) in highlighted,
                    "isAlternative": (u, v) in alternatives,
                }
                for u, v in edges
            ]

            return {"tree_meta": tree_meta, "nodes": node_list, "edges": edge_list}

        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.warning(f"WARNING FlowchartVisualizer.get_tree_json: {exc}")
            return None

    @staticmethod
    def visualize(results: dict, output_path: str | None = None) -> str | None:
        """Render a matplotlib PNG and return its file path.

        Parameters
        ----------
        results:     The full results dict from the analyser.
        output_path: Base path *without* extension.  A ".png" suffix is added.
                     When None a temporary file is used.

        Returns
        -------
        Absolute path to the saved PNG, or None on failure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
            import networkx as nx

            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except OSError:
                plt.style.use("seaborn-whitegrid")

            model_type = results.get("model_type", "")
            test_name  = results.get("test", results.get("test_name", model_type))
            alpha      = float(results.get("alpha", 0.05))

            nodes_info, edges, highlighted, alternatives, tree_meta = (
                FlowchartVisualizer._build_topology(results, model_type, alpha)
            )

            G = nx.DiGraph()
            for nid, info in nodes_info.items():
                G.add_node(nid, label=info["label"], pos=info["pos"])
            for u, v in edges:
                G.add_edge(u, v)

            pos_dict = {nid: info["pos"] for nid, info in nodes_info.items()}

            hl_edges  = [(u, v) for u, v in G.edges() if (u, v) in highlighted]
            alt_edges = [(u, v) for u, v in G.edges()
                         if (u, v) in alternatives and (u, v) not in highlighted]
            dim_edges = [(u, v) for u, v in G.edges()
                         if (u, v) not in highlighted and (u, v) not in alternatives]

            hl_nodes = {n for u, v in highlighted for n in (u, v)}

            # --- figure sizing ---
            xs = [xy[0] for xy in pos_dict.values()]
            ys = [xy[1] for xy in pos_dict.values()]
            x_span = (max(xs) - min(xs)) if xs else 4
            y_span = (max(ys) - min(ys)) if ys else 10
            fig_w = max(11.0, min(18.0, x_span * 2.8 + 9.0))
            fig_h = max(11.0, min(22.0, y_span * 1.2 + 4.0))

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_facecolor("#fafafa")

            sq_nodes = [n for n in G.nodes() if nodes_info[n]["isSquare"]]
            rd_nodes = [n for n in G.nodes() if not nodes_info[n]["isSquare"]]

            # Nodes
            for shape, nodelist in [("s", sq_nodes), ("o", rd_nodes)]:
                nx.draw_networkx_nodes(
                    G, pos_dict, ax=ax,
                    nodelist=[n for n in nodelist if n in hl_nodes],
                    node_shape=shape, node_size=3200,
                    node_color="#e6f3f2", edgecolors="#0f766e", linewidths=2.2,
                )
                nx.draw_networkx_nodes(
                    G, pos_dict, ax=ax,
                    nodelist=[n for n in nodelist if n not in hl_nodes],
                    node_shape=shape, node_size=3200,
                    node_color="white", edgecolors="#c5d3d7", linewidths=1.0,
                )

            # Edges
            _arrow_kw = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.0",
                             shrinkA=16, shrinkB=16)
            nx.draw_networkx_edges(
                G, pos_dict, ax=ax, edgelist=hl_edges,
                width=3.0, edge_color="#0f766e", arrows=True, arrowsize=20,
                connectionstyle="arc3,rad=0.0",
            )
            nx.draw_networkx_edges(
                G, pos_dict, ax=ax, edgelist=alt_edges,
                width=1.4, edge_color="#9fb8be", style="dashed",
                arrows=True, arrowsize=14,
                connectionstyle="arc3,rad=0.0",
            )
            nx.draw_networkx_edges(
                G, pos_dict, ax=ax, edgelist=dim_edges,
                width=0.9, edge_color="#d1d9dc", arrows=True, arrowsize=11,
                connectionstyle="arc3,rad=0.0",
            )

            # Labels
            nx.draw_networkx_labels(
                G, pos_dict, ax=ax,
                labels=nx.get_node_attributes(G, "label"),
                font_size=9.5, font_family="sans-serif", font_weight="normal",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.88, edgecolor="lightgray"),
            )

            # Legend
            legend_handles = [
                Line2D([0], [0], color="#0f766e", lw=3, label="Path taken"),
                Line2D([0], [0], color="#9fb8be", lw=1.5, linestyle="dashed",
                       label="Alternative (not used)"),
                mpatches.Patch(facecolor="#e6f3f2", edgecolor="#0f766e",
                               label="Active step"),
                mpatches.Patch(facecolor="white", edgecolor="#c5d3d7",
                               label="Inactive step"),
            ]
            ax.legend(handles=legend_handles, loc="upper right",
                      fontsize=9.5, frameon=True, facecolor="white",
                      framealpha=0.94)

            fig.suptitle(f"Analysis Workflow: {test_name}",
                         fontsize=13, y=0.995, weight="semibold")
            ax.axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.985])

            # Save
            if output_path:
                out_file = f"{output_path}.png"
                plt.savefig(out_file, format="png", dpi=180, facecolor="white",
                            bbox_inches="tight")
                plt.close("all")
                return out_file
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    plt.savefig(tmp.name, format="png", dpi=180, facecolor="white",
                                bbox_inches="tight")
                    path = tmp.name
                plt.close("all")
                return path

        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.info(f"FlowchartVisualizer.visualize error: {exc}")
            return None

    @staticmethod
    def generate_and_save(results: dict) -> str | None:
        """Thin wrapper matching the DecisionTreeVisualizer.generate_and_save API."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                base_path = tmp.name.replace(".png", "")
            return FlowchartVisualizer.visualize(results, output_path=base_path)
        except Exception as exc:
            logger.info(f"FlowchartVisualizer.generate_and_save error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Topology builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_topology(
        results: dict,
        model_type: str,
        alpha: float = 0.05,
    ) -> tuple[dict, set, set, set, dict]:
        """Build the node/edge topology for a given model type.

        Returns
        -------
        nodes_info  : {node_id: {label, pos:(x,y), isSquare}}
        edges       : set of (source, target)  — all edges in the graph
        highlighted : set of (source, target)  — the path actually taken
        alternatives: set of (source, target)  — non-taken variant edges
                      (rendered as dashed lines in both renderers)
        tree_meta   : dict with annotation metadata
        """
        try:
            from statistical_testing.validators import MIN_N_SMALL
        except ImportError:
            MIN_N_SMALL = 20

        # ── Common result fields ────────────────────────────────────────────
        p_value     = results.get("p_value", None)
        method      = str(results.get("method", "") or "").lower()
        n_samples   = int(results.get("n", 0) or 0)
        sig         = (p_value is not None and p_value < alpha)

        normality_check  = results.get("normality_check") or {}
        slope_hom        = results.get("slope_homogeneity") or {}
        both_normal      = normality_check.get("both_normal", None)
        slopes_ok: bool | None
        if slope_hom:
            slopes_ok = all(
                v.get("assumption_holds", True) for v in slope_hom.values()
            )
        else:
            slopes_ok = None
        # If simple slopes were computed, interaction was significant → slopes ≠ ok
        if results.get("simple_slopes_analysis") is not None or "simple_slopes" in results:
            slopes_ok = False

        # ── Sample-size tier (Correlation) ──────────────────────────────────
        if n_samples < MIN_N_SMALL:
            calculated_tier = f"Micro-Sample Mode (N < {MIN_N_SMALL})"
        elif n_samples < 100:
            calculated_tier = "Clinical Mode (20 ≤ N < 100)"
        else:
            calculated_tier = "Asymptotic Mode (N ≥ 100)"
        # Honour explicit tier stored in results
        calculated_tier = results.get("calculated_tier") or results.get("tier") or calculated_tier

        tree_meta = {
            "calculated_tier": calculated_tier,
            "total_samples":   n_samples,
        }

        # ── Spine constants ─────────────────────────────────────────────────
        # x = 0 is the vertical trunk; ±1.3 are the side-branch offsets.
        # y values run top → bottom (higher y = higher on screen, because the
        # Qt/JS renderers flip the y-axis: qy = (max_y - y) * SCALE_Y).

        nodes_info:  dict[str, dict] = {}
        edges:       set[tuple[str, str]] = set()
        highlighted: set[tuple[str, str]] = set()
        alternatives:set[tuple[str, str]] = set()

        # ====================================================================
        if model_type == "Correlation":
            r_val   = results.get("r", None)
            r_label = f"r = {r_val:.3f}" if r_val is not None else ""
            p_label = _fmt_p(p_value)
            sig_lbl = "Significant" if sig else "Not significant"

            used_pearson = (method == "pearson") or (both_normal is True)

            nodes_info = {
                "START":           {"label": "Start\nCorrelation Analysis",                        "pos": ( 0.0, 10.0), "isSquare": True},
                "TIER_MICRO":      {"label": f"Very small sample\n(N < {MIN_N_SMALL}) — interpret cautiously",  "pos": (-1.3,  8.5), "isSquare": True},
                "TIER_CLINICAL":   {"label": "Medium sample\n(20 ≤ N < 100) — standard approach",               "pos": ( 0.0,  8.5), "isSquare": True},
                "TIER_ASYMPTOTIC": {"label": "Large sample\n(N ≥ 100) — robust results expected",                "pos": ( 1.3,  8.5), "isSquare": True},
                "SKEW_KURT_CHECK": {"label": "Check if data is roughly\nnormally distributed",           "pos": ( 0.0,  7.0), "isSquare": True},
                "PEARSON":         {"label": "Normal data\n→ Pearson r",                                   "pos": (-1.3,  5.5), "isSquare": False},
                "SPEARMAN":        {"label": "Skewed data\n→ Spearman ρ",                                  "pos": ( 1.3,  5.5), "isSquare": False},
                "RESULT":          {"label": f"Is there a significant relationship?\n{r_label}  {p_label}\n{sig_lbl}", "pos": ( 0.0,  4.0), "isSquare": False},
                "CI":              {"label": "How precise is the estimate?\n(95% CI, Fisher z-transform)", "pos": (-1.3,  2.5), "isSquare": False},
                "EFFECT":          {"label": "How strong is the relationship?\n(|r|: ≥.1 small  ≥.3 med  ≥.5 large)", "pos": ( 1.3,  2.5), "isSquare": False},
            }
            edges = {
                ("START",           "TIER_MICRO"),
                ("START",           "TIER_CLINICAL"),
                ("START",           "TIER_ASYMPTOTIC"),
                ("TIER_MICRO",      "SPEARMAN"),
                ("TIER_CLINICAL",   "SKEW_KURT_CHECK"),
                ("TIER_ASYMPTOTIC", "SKEW_KURT_CHECK"),
                ("SKEW_KURT_CHECK", "PEARSON"),
                ("SKEW_KURT_CHECK", "SPEARMAN"),
                ("PEARSON",         "RESULT"),
                ("SPEARMAN",        "RESULT"),
                ("RESULT",          "CI"),
                ("RESULT",          "EFFECT"),
            }

            if n_samples < MIN_N_SMALL:
                highlighted.add(("START", "TIER_MICRO"))
                alternatives.update([("START", "TIER_CLINICAL"), ("START", "TIER_ASYMPTOTIC")])
                highlighted.add(("TIER_MICRO", "SPEARMAN"))
            else:
                tier_node  = "TIER_CLINICAL" if n_samples < 100 else "TIER_ASYMPTOTIC"
                other_tier = "TIER_ASYMPTOTIC" if n_samples < 100 else "TIER_CLINICAL"
                highlighted.add(("START", tier_node))
                alternatives.update([("START", "TIER_MICRO"), ("START", other_tier)])
                highlighted.add((tier_node, "SKEW_KURT_CHECK"))
                if used_pearson:
                    highlighted.add(("SKEW_KURT_CHECK", "PEARSON"))
                    alternatives.add(("SKEW_KURT_CHECK", "SPEARMAN"))
                    highlighted.add(("PEARSON", "RESULT"))
                else:
                    highlighted.add(("SKEW_KURT_CHECK", "SPEARMAN"))
                    alternatives.add(("SKEW_KURT_CHECK", "PEARSON"))
                    highlighted.add(("SPEARMAN", "RESULT"))
            highlighted.update([("RESULT", "CI"), ("RESULT", "EFFECT")])

        # ====================================================================
        elif model_type == "LinearRegression":
            r2       = results.get("r_squared", None)
            f_p      = results.get("f_p_value", None)
            cov_type = str(results.get("cov_type", "") or "").lower()
            is_hc3   = "hc3" in cov_type or cov_type == "robust"

            nodes_info = {
                "START":         {"label": "Start\nLinear Regression",                               "pos": ( 0.0, 10.0), "isSquare": True},
                "OLS_FIT":       {"label": "Fit a straight line\nthrough the data",                  "pos": ( 0.0,  8.5), "isSquare": True},
                "DIAGNOSTICS":   {"label": "Check if the model\nassumptions are met",                "pos": ( 0.0,  7.0), "isSquare": True},
                "ROBUST_BRANCH": {"label": "Are the prediction errors\nevenly spread?",               "pos": ( 0.0,  5.5), "isSquare": True},
                "COV_HC3":       {"label": "Uneven spread detected\n→ adjusted estimation",          "pos": (-1.3,  4.0), "isSquare": False},
                "COV_NONROBUST": {"label": "Even spread confirmed\n→ standard estimation",           "pos": ( 1.3,  4.0), "isSquare": False},
                "COEFFICIENTS":  {"label": "How much does each predictor contribute?\n(β, SE, t, p, 95% CI)", "pos": ( 0.0,  2.5), "isSquare": False},
                "MODEL_FIT":     {"label": f"How well does the model fit the data?\n{_fmt_r2(r2)}  {_fmt_p(f_p)}", "pos": (-1.3,  1.0), "isSquare": False},
                "EFFECT":        {"label": "How much variance is explained?\n(R²: ≥.01 small  ≥.09 med  ≥.25 large)", "pos": ( 1.3,  1.0), "isSquare": False},
            }
            edges = {
                ("START",         "OLS_FIT"),
                ("OLS_FIT",       "DIAGNOSTICS"),
                ("DIAGNOSTICS",   "ROBUST_BRANCH"),
                ("ROBUST_BRANCH", "COV_HC3"),
                ("ROBUST_BRANCH", "COV_NONROBUST"),
                ("COV_HC3",       "COEFFICIENTS"),
                ("COV_NONROBUST", "COEFFICIENTS"),
                ("COEFFICIENTS",  "MODEL_FIT"),
                ("COEFFICIENTS",  "EFFECT"),
            }
            highlighted = {
                ("START", "OLS_FIT"),
                ("OLS_FIT", "DIAGNOSTICS"),
                ("DIAGNOSTICS", "ROBUST_BRANCH"),
            }
            if is_hc3:
                highlighted.add(("ROBUST_BRANCH", "COV_HC3"))
                highlighted.add(("COV_HC3", "COEFFICIENTS"))
                alternatives.add(("ROBUST_BRANCH", "COV_NONROBUST"))
            else:
                highlighted.add(("ROBUST_BRANCH", "COV_NONROBUST"))
                highlighted.add(("COV_NONROBUST", "COEFFICIENTS"))
                alternatives.add(("ROBUST_BRANCH", "COV_HC3"))
            highlighted.update([("COEFFICIENTS", "MODEL_FIT"), ("COEFFICIENTS", "EFFECT")])

        # ====================================================================
        elif model_type == "LogisticRegression":
            auc         = results.get("effect_size", None)
            if auc is None:
                auc = (results.get("roc_data") or {}).get("auc", None)
            auc_label   = f"AUC = {auc:.3f}" if auc is not None else ""
            is_firth    = (str(results.get("model_variant", "") or "").lower() == "firth penalized likelihood"
                           or results.get("method_name", "") == "Firth")

            nodes_info = {
                "START":                {"label": "Start\nLogistic Regression",                                "pos": ( 0.0, 10.0), "isSquare": True},
                "SEPARATION_CHECK":     {"label": "Check if the model can be\nreliably estimated",              "pos": ( 0.0,  8.5), "isSquare": True},
                "STANDARD_LOGIT":       {"label": "Standard method\n(data looks clean)",                       "pos": (-1.3,  7.0), "isSquare": False},
                "FIRTH_REGRESSION":     {"label": "Safer method\n(small or separated groups)",                 "pos": ( 1.3,  7.0), "isSquare": False},
                "CALIBRATION_ANALYSIS": {"label": "How accurate and reliable\nare the predictions?",           "pos": ( 0.0,  5.5), "isSquare": True},
                "SIG_YES":              {"label": "Which factors matter?",                                       "pos": (-1.3,  4.0), "isSquare": False},
                "SIG_NO":               {"label": "No meaningful predictors found",                             "pos": ( 1.3,  4.0), "isSquare": False},
                "OR_TABLE":             {"label": "How strongly does each factor\naffect the outcome? (OR, 95% CI, p)", "pos": (-2.0,  2.5), "isSquare": False},
                "ROC_AUC":              {"label": f"How well does the model separate\nthe two groups? {auc_label}", "pos": ( 0.0,  2.5), "isSquare": False},
                "AIC_BIC":              {"label": "Is this model better\nthan a simpler one? (AIC / BIC)",       "pos": ( 2.0,  2.5), "isSquare": False},
            }
            edges = {
                ("START",                "SEPARATION_CHECK"),
                ("SEPARATION_CHECK",     "STANDARD_LOGIT"),
                ("SEPARATION_CHECK",     "FIRTH_REGRESSION"),
                ("STANDARD_LOGIT",       "CALIBRATION_ANALYSIS"),
                ("FIRTH_REGRESSION",     "CALIBRATION_ANALYSIS"),
                ("CALIBRATION_ANALYSIS", "SIG_YES"),
                ("CALIBRATION_ANALYSIS", "SIG_NO"),
                ("SIG_YES",              "OR_TABLE"),
                ("SIG_YES",              "ROC_AUC"),
                ("SIG_YES",              "AIC_BIC"),
                ("SIG_NO",               "ROC_AUC"),
                ("SIG_NO",               "AIC_BIC"),
            }
            highlighted = {("START", "SEPARATION_CHECK")}
            if is_firth:
                highlighted.update([("SEPARATION_CHECK", "FIRTH_REGRESSION"),
                                     ("FIRTH_REGRESSION", "CALIBRATION_ANALYSIS")])
                alternatives.add(("SEPARATION_CHECK", "STANDARD_LOGIT"))
            else:
                highlighted.update([("SEPARATION_CHECK", "STANDARD_LOGIT"),
                                     ("STANDARD_LOGIT", "CALIBRATION_ANALYSIS")])
                alternatives.add(("SEPARATION_CHECK", "FIRTH_REGRESSION"))
            if sig:
                highlighted.update([("CALIBRATION_ANALYSIS", "SIG_YES"),
                                     ("SIG_YES", "OR_TABLE"),
                                     ("SIG_YES", "ROC_AUC"),
                                     ("SIG_YES", "AIC_BIC")])
                alternatives.add(("CALIBRATION_ANALYSIS", "SIG_NO"))
            else:
                highlighted.update([("CALIBRATION_ANALYSIS", "SIG_NO"),
                                     ("SIG_NO", "ROC_AUC"),
                                     ("SIG_NO", "AIC_BIC")])
                alternatives.add(("CALIBRATION_ANALYSIS", "SIG_YES"))

        # ====================================================================
        elif model_type == "ANCOVA":
            eta2        = results.get("effect_size", None)
            eta2_label  = f"η² = {eta2:.3f}" if eta2 is not None else ""

            nodes_info = {
                "START":            {"label": "Start\nANCOVA Analysis",                                   "pos": ( 0.0, 10.0), "isSquare": True},
                "COVAR":            {"label": "Set up groups\nand control variables",                          "pos": ( 0.0,  8.5), "isSquare": True},
                "SLOPE_HOM":        {"label": f"Check if the control variable works\nequally across groups (α={alpha})", "pos": ( 0.0,  7.0), "isSquare": True},
                "SLOPE_OK":         {"label": "Control variable works equally\n→ ANCOVA valid",               "pos": (-1.3,  5.5), "isSquare": False},
                "SLOPE_FAIL":       {"label": "Control variable behaves differently\nacross groups",           "pos": ( 1.3,  5.5), "isSquare": False},
                "SIMPLE_SLOPES_JN": {"label": "Explore where group differences\nemerge",                     "pos": ( 1.3,  4.0), "isSquare": True},
                "FIT":              {"label": f"Does the model explain the data well?\n{_fmt_p(p_value)}  {eta2_label}", "pos": ( 0.0,  2.5), "isSquare": True},
                "SIG_YES":          {"label": "Groups differ after controlling\nfor the covariate",           "pos": (-1.3,  1.0), "isSquare": False},
                "SIG_NO":           {"label": "No group differences found",                                    "pos": ( 1.3,  1.0), "isSquare": False},
                "POSTHOC":          {"label": "Which specific groups differ\nfrom each other?",               "pos": (-1.3, -0.5), "isSquare": False},
                "EFFECT":           {"label": "How large is the group difference?\n(Partial η²: ≥.01 small  ≥.06 med  ≥.14 large)", "pos": ( 1.3, -0.5), "isSquare": False},
            }
            edges = {
                ("START",           "COVAR"),
                ("COVAR",           "SLOPE_HOM"),
                ("SLOPE_HOM",       "SLOPE_OK"),
                ("SLOPE_HOM",       "SLOPE_FAIL"),
                ("SLOPE_OK",        "FIT"),
                ("SLOPE_FAIL",      "SIMPLE_SLOPES_JN"),
                ("SIMPLE_SLOPES_JN","FIT"),
                ("FIT",             "SIG_YES"),
                ("FIT",             "SIG_NO"),
                ("SIG_YES",         "POSTHOC"),
                ("SIG_YES",         "EFFECT"),
                ("SIG_NO",          "EFFECT"),
            }
            highlighted = {("START", "COVAR"), ("COVAR", "SLOPE_HOM")}
            if slopes_ok is False:
                highlighted.update([("SLOPE_HOM", "SLOPE_FAIL"),
                                     ("SLOPE_FAIL", "SIMPLE_SLOPES_JN"),
                                     ("SIMPLE_SLOPES_JN", "FIT")])
                alternatives.add(("SLOPE_HOM", "SLOPE_OK"))
            else:
                highlighted.update([("SLOPE_HOM", "SLOPE_OK"), ("SLOPE_OK", "FIT")])
                alternatives.add(("SLOPE_HOM", "SLOPE_FAIL"))
            if sig:
                highlighted.update([("FIT", "SIG_YES"), ("SIG_YES", "POSTHOC"), ("SIG_YES", "EFFECT")])
                alternatives.add(("FIT", "SIG_NO"))
            else:
                highlighted.update([("FIT", "SIG_NO"), ("SIG_NO", "EFFECT")])
                alternatives.add(("FIT", "SIG_YES"))

        # ====================================================================
        elif model_type == "LMM":
            icc_val   = results.get("icc", results.get("effect_size", None))
            icc_label = f"ICC = {icc_val:.3f}" if icc_val is not None else ""
            df_method = str(results.get("df_method", "") or "")
            is_bw     = ("Between" in df_method) or (n_samples > 0 and n_samples < 100)

            nodes_info = {
                "START":                {"label": "Start\nLinear Mixed Model (LMM)",                              "pos": ( 0.0, 10.0), "isSquare": True},
                "RANDOM_STRUCTURE_LRT": {"label": "Find the best model structure\nfor the nested data",            "pos": ( 0.0,  8.5), "isSquare": True},
                "FINAL_FIT":            {"label": "Estimate the model parameters\n(REML)",                        "pos": ( 0.0,  7.0), "isSquare": True},
                "BW_DF_CORRECTION":     {"label": "Small sample → conservative\nsignificance testing (N < 100)", "pos": (-1.3,  5.5), "isSquare": True},
                "ASYMP_DF":             {"label": "Large sample → standard\nsignificance testing (N ≥ 100)",   "pos": ( 1.3,  5.5), "isSquare": True},
                "FIXED_EFFECTS_TEST":   {"label": f"Do the predictors have\na consistent effect? {_fmt_p(p_value)}", "pos": ( 0.0,  4.0), "isSquare": True},
                "SIG_YES":              {"label": "Yes — meaningful effects found",                               "pos": (-1.3,  2.5), "isSquare": False},
                "SIG_NO":               {"label": "No consistent effects found",                                  "pos": ( 1.3,  2.5), "isSquare": False},
                "FIXEF_TABLE":          {"label": "Detailed results for each predictor\n(β, SE, t/z, p, 95% CI)", "pos": (-2.0,  1.0), "isSquare": False},
                "ICC_VAL":              {"label": f"How similar are observations\nwithin the same group? {icc_label}", "pos": ( 0.0,  1.0), "isSquare": False},
                "MODEL_FIT":            {"label": "Compare this model to alternatives\n(AIC / BIC)",              "pos": ( 2.0,  1.0), "isSquare": False},
            }
            edges = {
                ("START",                "RANDOM_STRUCTURE_LRT"),
                ("RANDOM_STRUCTURE_LRT", "FINAL_FIT"),
                ("FINAL_FIT",            "BW_DF_CORRECTION"),
                ("FINAL_FIT",            "ASYMP_DF"),
                ("BW_DF_CORRECTION",     "FIXED_EFFECTS_TEST"),
                ("ASYMP_DF",             "FIXED_EFFECTS_TEST"),
                ("FIXED_EFFECTS_TEST",   "SIG_YES"),
                ("FIXED_EFFECTS_TEST",   "SIG_NO"),
                ("SIG_YES",              "FIXEF_TABLE"),
                ("SIG_YES",              "ICC_VAL"),
                ("SIG_NO",               "ICC_VAL"),
                ("SIG_NO",               "MODEL_FIT"),
                ("ICC_VAL",              "MODEL_FIT"),
            }
            highlighted = {
                ("START", "RANDOM_STRUCTURE_LRT"),
                ("RANDOM_STRUCTURE_LRT", "FINAL_FIT"),
            }
            if is_bw:
                highlighted.update([("FINAL_FIT", "BW_DF_CORRECTION"),
                                     ("BW_DF_CORRECTION", "FIXED_EFFECTS_TEST")])
                alternatives.add(("FINAL_FIT", "ASYMP_DF"))
            else:
                highlighted.update([("FINAL_FIT", "ASYMP_DF"),
                                     ("ASYMP_DF", "FIXED_EFFECTS_TEST")])
                alternatives.add(("FINAL_FIT", "BW_DF_CORRECTION"))
            if sig:
                highlighted.update([("FIXED_EFFECTS_TEST", "SIG_YES"),
                                     ("SIG_YES", "FIXEF_TABLE"),
                                     ("SIG_YES", "ICC_VAL")])
                alternatives.add(("FIXED_EFFECTS_TEST", "SIG_NO"))
            else:
                highlighted.update([("FIXED_EFFECTS_TEST", "SIG_NO"),
                                     ("SIG_NO", "ICC_VAL"),
                                     ("SIG_NO", "MODEL_FIT")])
                alternatives.add(("FIXED_EFFECTS_TEST", "SIG_YES"))
            highlighted.add(("ICC_VAL", "MODEL_FIT"))

        # ====================================================================
        else:  # CorrelationMatrix (and any unrecognised type)
            method_label      = str(results.get("method", "Auto") or "Auto").capitalize()
            correction_label  = str(results.get("correction", "None") or "None")
            variables         = results.get("variables", []) or []
            n_vars            = len(variables)

            nodes_info = {
                "START":    {"label": f"Start\nExploratory Correlation Matrix\n({n_vars} variables)", "pos": ( 0.0, 6.0), "isSquare": True},
                "METHOD":   {"label": f"Method chosen automatically\nbased on data distribution ({method_label})", "pos": ( 0.0, 4.5), "isSquare": True},
                "CORRECT":  {"label": f"Multiple Testing Correction\n{correction_label}",                       "pos": ( 0.0, 3.0), "isSquare": True},
                "MATRIX":   {"label": "Correlation Matrix\n(r / ρ per pair)",                                   "pos": (-1.3, 1.5), "isSquare": False},
                "SIG_MAP":  {"label": "Significance Map\n(corrected p-values)",                                 "pos": ( 1.3, 1.5), "isSquare": False},
                "INTERPRET":{"label": "Which relationships are meaningful\nand in which direction?",             "pos": ( 0.0, 0.0), "isSquare": False},
            }
            edges = {
                ("START",   "METHOD"),
                ("METHOD",  "CORRECT"),
                ("CORRECT", "MATRIX"),
                ("CORRECT", "SIG_MAP"),
                ("MATRIX",  "INTERPRET"),
                ("SIG_MAP", "INTERPRET"),
            }
            highlighted = {
                ("START",   "METHOD"),
                ("METHOD",  "CORRECT"),
                ("CORRECT", "MATRIX"),
                ("CORRECT", "SIG_MAP"),
                ("MATRIX",  "INTERPRET"),
                ("SIG_MAP", "INTERPRET"),
            }
            # CorrelationMatrix has no mutually-exclusive branches → no alternatives

        return nodes_info, edges, highlighted, alternatives, tree_meta

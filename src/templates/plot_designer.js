(function () {
  var plotNode = document.getElementById("pd-plot");
  if (!plotNode || typeof Plotly === "undefined") {
    return;
  }

  function parseJsonNode(id, fallback) {
    var node = document.getElementById(id);
    if (!node) return fallback;
    try {
      return JSON.parse(node.textContent || "");
    } catch (error) {
      return fallback;
    }
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function stableJitter(baseValue, pointIndex, groupIndex, amplitude) {
    var seed = (pointIndex + 1) * 12.9898 + (groupIndex + 1) * 78.233;
    var pseudo = Math.sin(seed) * 43758.5453;
    var rand01 = pseudo - Math.floor(pseudo);
    return baseValue + (rand01 - 0.5) * amplitude;
  }

  var plotData = parseJsonNode("pd-data-plot", {});
  var subjectTrajectories = parseJsonNode("pd-data-subject-trajectories", []);
  var referenceLinesPayload = parseJsonNode("pd-data-reference-lines", []);
  var plotStats = parseJsonNode("pd-data-stats-summary", null);
  if (!plotStats || typeof plotStats !== "object") {
    plotStats = parseJsonNode("pd-data-stats", {});
  }
  var pairwiseData = parseJsonNode("pd-data-pairs", []);
  var groupOrder = parseJsonNode("pd-data-order", []);
  var groupFactorMapPayload = parseJsonNode("pd-data-group-factor-map", {});

  // Shared style tokens injected by html_exporter.py from visualization/style_tokens.py.
  // Each field falls back to the historical literal so an older report still renders.
  // These are DEFAULTS; the user can still change palette / colours interactively.
  var styleTokens = parseJsonNode("pd-data-style", {});
  function _numOr(v, d) { return (typeof v === "number" && isFinite(v)) ? v : d; }
  var plotStyle = {
    palettes: (styleTokens.palettes && typeof styleTokens.palettes === "object") ? styleTokens.palettes : {
      Nature:  ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"],
      Science: ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#999999"],
      NEJM:    ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91"],
      Lancet:  ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A"],
      Tab10:   ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    },
    defaultPalette: styleTokens.default_palette || "grayscale",
    grayscaleFloor: styleTokens.grayscale_floor || "#404040",
    pointFillColor: styleTokens.point_fill_color || "#000000",
    pointEdgeColor: styleTokens.point_edge_color || "#000000",
    pointEdgeWidth: _numOr(styleTokens.point_edge_width, 1),
    pointSize: _numOr(styleTokens.point_size, 6),
    shapeOutlineColor: styleTokens.shape_outline_color || "#000000",
    shapeOutlineWidth: _numOr(styleTokens.shape_outline_width, 2),
    frameColor: styleTokens.frame_color || "rgba(22,49,58,0.75)",
    frameLinewidth: _numOr(styleTokens.frame_linewidth, 0.7)
  };
  // The frame styling below used to be copy-pasted into every axis of every
  // plot type (Bar/Box/Violin, Raincloud, Forest, plus the value axis), so a
  // change had to be made in six places and in practice never was — that is how
  // the x-label clipping survived several fixes. Build every axis through this.
  // tickMode/axisMirror are per-render locals, so they are passed in.
  function axisFrame(tickMode, axisMirror) {
    return {
      showline: true,
      linecolor: plotStyle.frameColor,
      linewidth: Math.max(0.5, state.axisThickness),
      ticks: tickMode,
      tickwidth: Math.max(0.5, state.axisThickness),
      ticklen: Math.max(4, Math.round(4 + state.axisThickness * 2)),
      mirror: axisMirror
    };
  }

  function grayFloorChannel() {
    var v = parseInt(String(plotStyle.grayscaleFloor).replace("#", "").slice(0, 2), 16);
    return isFinite(v) ? v : 64;
  }
  // Data points: always solid black (fill + edge), independent of the palette.
  function pointEdge() {
    return { width: plotStyle.pointEdgeWidth, color: plotStyle.pointEdgeColor };
  }

  function normalizeReferenceLines(rawLines) {
    if (!Array.isArray(rawLines)) {
      return [];
    }

    var normalized = [];
    rawLines.forEach(function (line, index) {
      var value = null;
      var label = "Threshold " + String(index + 1);
      var dash = "dash";
      var color = "rgba(159,58,56,0.82)";
      var width = 1.5;

      if (typeof line === "number" && Number.isFinite(line)) {
        value = line;
      } else if (line && typeof line === "object") {
        var numericCandidate = Number(line.value);
        if (!Number.isFinite(numericCandidate)) {
          numericCandidate = Number(line.y);
        }
        if (!Number.isFinite(numericCandidate)) {
          numericCandidate = Number(line.threshold);
        }
        if (!Number.isFinite(numericCandidate)) {
          return;
        }
        value = numericCandidate;

        if (typeof line.label === "string" && line.label.trim()) {
          label = line.label.trim();
        } else if (typeof line.name === "string" && line.name.trim()) {
          label = line.name.trim();
        }

        var dashCandidate = String(line.dash || "").trim().toLowerCase();
        if (["solid", "dash", "dot", "dashdot"].indexOf(dashCandidate) !== -1) {
          dash = dashCandidate;
        }

        if (typeof line.color === "string" && line.color.trim()) {
          color = line.color.trim();
        }

        var widthCandidate = Number(line.width);
        if (Number.isFinite(widthCandidate)) {
          width = Math.min(4, Math.max(0.6, widthCandidate));
        }
      } else {
        return;
      }

      normalized.push({
        value: value,
        label: label,
        dash: dash,
        color: color,
        width: width
      });
    });

    if (normalized.length > 30) {
      normalized = normalized.slice(0, 30);
    }
    return normalized;
  }

  var thresholdReferenceLines = normalizeReferenceLines(referenceLinesPayload);

  if (!Array.isArray(groupOrder) || !groupOrder.length) {
    groupOrder = Object.keys(plotData || {});
  }
  groupOrder = groupOrder.filter(function (group) {
    return Array.isArray(plotData[group]) && plotData[group].length > 0;
  });

  if (!groupOrder.length) {
    var warning = document.getElementById("pd-warning");
    if (warning) {
      warning.textContent = "Designer disabled: no valid group data found in this export.";
    }
    return;
  }

  var missingStats = groupOrder.filter(function (group) {
    var stats = plotStats[group] || {};
    return !isFiniteNumber(stats.mean) || !isFiniteNumber(stats.min) || !isFiniteNumber(stats.max);
  });

  if (missingStats.length) {
    var statsWarning = document.getElementById("pd-warning");
    if (statsWarning) {
      statsWarning.textContent = "Designer disabled: immutable statistics summary missing for one or more groups.";
    }
    return;
  }

  // Curated colour palettes, selectable via Style > Colors > "Palette".
  // These mirror the desktop app's journal palettes (datavisualizer.py /
  // plot_aesthetics_dialog.py) so the HTML report and the app agree. The
  // default is a grayscale ramp (like the desktop "Greys" default), generated
  // black -> white across however many groups the design has, so a t-test
  // (2 groups) and an ANOVA (>2 groups) both look right.
  // Palettes come from the shared source (visualization/style_tokens.py) via the
  // injected pd-data-style blob; plotStyle.palettes carries a fallback copy.
  var PALETTES = plotStyle.palettes;
  var DEFAULT_PALETTE_NAME = plotStyle.defaultPalette;
  // Evenly spaced greys from the shared floor (not pure black, so black data
  // points stay legible on the darkest segment) up to white, for n groups.
  function grayscaleRamp(n) {
    var floor = grayFloorChannel();
    var ceil = 255;
    if (n <= 1) { var hf = ("0" + floor.toString(16)).slice(-2); return ["#" + hf + hf + hf]; }
    var out = [];
    for (var i = 0; i < n; i++) {
      var v = Math.round(floor + (ceil - floor) * i / (n - 1));
      var h = ("0" + v.toString(16)).slice(-2);
      out.push("#" + h + h + h);
    }
    return out;
  }
  // "grayscale" is generated per group count; every other name is a fixed list.
  function resolvePalette(name, n) {
    if (name === "grayscale") return grayscaleRamp(n);
    return PALETTES[name] || grayscaleRamp(n);
  }
  // Outline for filled shapes (bar/box/violin/raincloud): black, with a
  // user-adjustable width, applied uniformly so a white/light fill always has a
  // visible border on every plot type.
  function shapeOutline() {
    return { color: plotStyle.shapeOutlineColor, width: state.outlineWidth };
  }
  // Prioritize combinations that remain separable in dense grayscale exports.
  var defaultPatternCycle = ["x", "\\", "/", "-", "|", "+", "."];
  // Default point symbol is a circle for every group. Other shapes stay
  // available per group via the Style > Colors symbol dropdowns; circle is
  // listed first so the fallback is also a circle.
  var defaultSymbolCycle = ["circle", "square", "diamond", "cross", "triangle-up"];
  var fontStacks = {
    "Arial": 'Arial, "Helvetica Neue", Helvetica, sans-serif',
    "Helvetica": '"Helvetica Neue", Helvetica, Arial, sans-serif',
    "Calibri": 'Calibri, "Segoe UI", Arial, sans-serif',
    "Segoe UI": '"Segoe UI", "Helvetica Neue", Arial, sans-serif',
    "Avenir Next": '"Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif',
    "Times New Roman": '"Times New Roman", Times, serif',
    "Cambria": 'Cambria, Georgia, "Times New Roman", serif',
    "Georgia": 'Georgia, "Times New Roman", serif',
    "Garamond": 'Garamond, "Palatino Linotype", serif',
    "Palatino Linotype": '"Palatino Linotype", Palatino, "Times New Roman", serif'
  };

  function resolveFontFamilyStack(fontName) {
    if (!fontName) {
      return fontStacks.Arial;
    }
    return fontStacks[fontName] || ('"' + fontName + '", ' + fontStacks.Arial);
  }

  function isFontAvailable(fontName) {
    if (!fontName) return true;
    if (typeof document !== "undefined" && document.fonts && typeof document.fonts.check === "function") {
      try {
        if (document.fonts.check('12px "' + fontName + '"')) {
          return true;
        }
      } catch (error) {
        // Fallback to canvas-based heuristic below.
      }
    }

    if (!isFontAvailable._canvas) {
      isFontAvailable._canvas = document.createElement("canvas");
    }
    var ctx = isFontAvailable._canvas.getContext("2d");
    if (!ctx) return true;
    var sample = "abcdefghijklmnopqrstuvwxyz0123456789";
    ctx.font = "16px monospace";
    var baseWidth = ctx.measureText(sample).width;
    ctx.font = '16px "' + fontName + '", monospace';
    var testWidth = ctx.measureText(sample).width;
    return Math.abs(testWidth - baseWidth) > 0.1;
  }

  function updateFontPreviewStatus() {
    var fontWarning = document.getElementById("pd-font-warning");
    if (!fontWarning) return;

    if (isFontAvailable(state.fontFamily)) {
      fontWarning.textContent = "";
      return;
    }

    fontWarning.textContent = 'Selected font is not available in this browser; preview uses a fallback.';
  }

  function styleFontSelectOptions() {
    var select = document.getElementById("pd-font-family");
    if (!select) return;
    Array.from(select.options).forEach(function (optionNode) {
      var family = optionNode.value || optionNode.textContent || "Arial";
      optionNode.style.fontFamily = resolveFontFamilyStack(family);
    });
  }

  var state = {
    plotType: "Bar",
    title: "",
    xLabel: "Groups",
    yLabel: "Values",
    fontFamily: "Arial",
    titleSize: 16,
    axisSize: 12,
    alpha: 0.85,
    outlineWidth: plotStyle.shapeOutlineWidth,
    showPoints: true,
    showErrorBars: true,
    centralMeasure: "mean",
    errorType: "sd",
    errorDirection: "both",
    logX: false,
    logY: false,
    minorTicks: false,
    gridStyle: "none",
    gridAlpha: 0.3,
    axisThickness: 0.7,
    tickDirection: "out",
    xTickAngle: 0,
    yAxisFormat: "auto",
    yMin: null,
    yMax: null,
    showZeroReferenceLine: false,
    showUnitReferenceLine: false,
    showThresholdReferenceLines: thresholdReferenceLines.length > 0,
    referenceLineDash: "dash",
    referenceLineWidth: 1.5,
    showLegend: true,
    legendOrientation: "v",
    legendX: 1.02,
    legendY: 1.0,
    legendXAnchor: "left",
    legendYAnchor: "top",
    significanceMode: "brackets",
    significanceLineWidth: 1.7,
    significanceSpacingScale: 1.0,
    significanceStarSize: 14,
    significanceStarOffset: 2,
    exportWidth: 8,
    exportHeight: 6,
    pngScale: 3,
    colors: {},
    patterns: {},
    symbols: {},
    autoPatternsEnabled: false,
    visiblePairIds: [],
    groupLabels: {},
    pointLayout: "jitter",
    grouping: {
      enabled: false,
      map: {},
      majorOrder: [],
      minorOrder: [],
      mode: "traces"
    }
  };

  function hasUsableSubjectTrajectories() {
    if (!Array.isArray(subjectTrajectories) || !subjectTrajectories.length) {
      return false;
    }
    return subjectTrajectories.some(function (trajectory) {
      return trajectory && Array.isArray(trajectory.points) && trajectory.points.length >= 2;
    });
  }

  function updatePairedLineControlState() {
    // Paired-lines controls were removed from the UI; nothing to sync.
  }

  function setControlDisabled(controlId, disabled) {
    var control = document.getElementById(controlId);
    if (!control) return;
    control.disabled = disabled;
    var wrapper = control.closest(".pd-row, .pd-check");
    if (wrapper) {
      wrapper.classList.toggle("is-disabled", disabled);
    }
  }

  function updateReferenceNote() {
    var noteNode = document.getElementById("pd-ref-note");
    if (!noteNode) return;
    if (state.plotType === "Raincloud") {
      noteNode.textContent = "Reference lines are disabled for Raincloud layout.";
      return;
    }
    if (thresholdReferenceLines.length > 0) {
      noteNode.textContent = String(thresholdReferenceLines.length) + " threshold line(s) from payload.";
    } else {
      noteNode.textContent = "No thresholds found in result payload.";
    }
  }

  // Letters are refused when the post-hoc did not compare every pair, and the
  // pair checkboxes are meaningless in letters mode (see buildLetters). Both are
  // explained rather than silently removed -- a greyed-out control with a reason
  // teaches; a missing one confuses.
  function updateSignificanceAvailability() {
    var noteNode = document.getElementById("pd-significance-note");
    var lettersOption = document.querySelector("#pd-significance-mode option[value=\"letters\"]");
    var pairRoot = document.getElementById("pd-pair-controls");
    var support = lettersSupported(pairsForPlot(groupIndexMap()));
    var lettersUsable = support.ok && state.plotType !== "Forest";

    if (lettersOption) lettersOption.disabled = !lettersUsable;
    if (!lettersUsable && state.significanceMode === "letters") {
      state.significanceMode = "brackets";
      setSelect("pd-significance-mode", "brackets");
    }

    var lettersActive = state.significanceMode === "letters";
    if (pairRoot) {
      pairRoot.classList.toggle("is-disabled", lettersActive);
      Array.from(pairRoot.querySelectorAll(".pd-pair-toggle")).forEach(function (node) {
        node.disabled = lettersActive;
      });
    }

    if (!noteNode) return;
    if (state.plotType === "Forest") {
      noteNode.textContent = "Forest rows are contrasts, not groups \u2014 no significance layer applies.";
    } else if (!support.ok) {
      noteNode.textContent = support.reason;
    } else if (lettersActive) {
      noteNode.textContent = "Groups sharing a letter are not significantly different. "
        + "Letters use all comparisons, so individual pairs cannot be hidden.";
    } else {
      noteNode.textContent = "";
    }
  }

  function applyPlotTypeVisibility() {
    var currentType = state.plotType;
    var elements = document.querySelectorAll("[data-plot-types]");
    for (var i = 0; i < elements.length; i++) {
      var el = elements[i];
      var allowedTypes = el.dataset.plotTypes.split(",");
      var isVisible = false;
      for (var j = 0; j < allowedTypes.length; j++) {
        if (allowedTypes[j].trim() === currentType) {
          isVisible = true;
          break;
        }
      }
      if (isVisible) {
        el.classList.remove("pd-hidden-by-type");
      } else {
        el.classList.add("pd-hidden-by-type");
      }
    }
  }

  // A plot type that cannot render a setting has it switched off, so the value
  // never reaches a builder that would choke on it — but the user's choice is
  // REMEMBERED and restored as soon as they return to a type that supports it.
  // Before this, one visit to Forest permanently wiped the log axes, reference
  // lines, y-limits and significance brackets: switching back left every box
  // unchecked and the user had to redo the whole configuration. The same trap
  // had already been fixed once for error bars (see below); this generalises it.
  var hiddenControlStash = {};

  function suspendWhileUnsupported(name, supported, capture, clear, restore) {
    if (!supported) {
      if (!(name in hiddenControlStash)) {
        hiddenControlStash[name] = capture();
      }
      clear();
    } else if (name in hiddenControlStash) {
      restore(hiddenControlStash[name]);
      delete hiddenControlStash[name];
    }
  }

  function setChecked(id, value) {
    var el = document.getElementById(id);
    if (el) el.checked = !!value;
  }

  function setSelect(id, value) {
    var el = document.getElementById(id);
    if (el) el.value = String(value);
  }

  function setValue(id, value) {
    var el = document.getElementById(id);
    if (el) el.value = (value === null || value === undefined) ? "" : String(value);
  }

  function resetStateForHiddenControls() {
    var type = state.plotType;
    var barBoxViolin = ["Bar", "Box", "Violin"];
    var barBoxViolinRaincloud = ["Bar", "Box", "Violin", "Raincloud"];

    // Error bars and auto-pattern render only for Bar (the trace builders guard
    // on plotType), so we deliberately do NOT force them off here — doing that
    // silently dropped the user's choice when switching type and back.

    // Reference lines: Bar, Box, Violin only
    suspendWhileUnsupported("referenceLines", barBoxViolin.indexOf(type) !== -1,
      function () {
        return {
          zero: state.showZeroReferenceLine,
          unit: state.showUnitReferenceLine,
          thresholds: state.showThresholdReferenceLines
        };
      },
      function () {
        state.showZeroReferenceLine = false;
        state.showUnitReferenceLine = false;
        state.showThresholdReferenceLines = false;
        ["pd-ref-zero", "pd-ref-unit", "pd-ref-thresholds"].forEach(function (id) {
          setChecked(id, false);
        });
      },
      function (saved) {
        state.showZeroReferenceLine = saved.zero;
        state.showUnitReferenceLine = saved.unit;
        state.showThresholdReferenceLines = saved.thresholds;
        setChecked("pd-ref-zero", saved.zero);
        setChecked("pd-ref-unit", saved.unit);
        setChecked("pd-ref-thresholds", saved.thresholds);
      });

    // Grouping: Bar, Box, Violin, Raincloud only
    suspendWhileUnsupported("grouping", barBoxViolinRaincloud.indexOf(type) !== -1,
      function () { return { enabled: state.grouping.enabled }; },
      function () {
        state.grouping.enabled = false;
        setChecked("pd-group-enabled", false);
      },
      function (saved) {
        state.grouping.enabled = saved.enabled;
        setChecked("pd-group-enabled", saved.enabled);
      });

    // Log axes + Y range/format: Bar, Box, Violin, Raincloud only
    suspendWhileUnsupported("axisScale", barBoxViolinRaincloud.indexOf(type) !== -1,
      function () {
        return { logX: state.logX, logY: state.logY, yMin: state.yMin, yMax: state.yMax };
      },
      function () {
        state.logX = false;
        state.logY = false;
        state.yMin = null;
        state.yMax = null;
        setChecked("pd-log-x", false);
        setChecked("pd-log-y", false);
        setValue("pd-y-min", "");
        setValue("pd-y-max", "");
      },
      function (saved) {
        state.logX = saved.logX;
        state.logY = saved.logY;
        state.yMin = saved.yMin;
        state.yMax = saved.yMax;
        setChecked("pd-log-x", saved.logX);
        setChecked("pd-log-y", saved.logY);
        setValue("pd-y-min", saved.yMin);
        setValue("pd-y-max", saved.yMax);
      });

    // Forest ignores showPoints in its builder, so leave the user's choice
    // intact across switches; only the significance layer is not drawn there.
    suspendWhileUnsupported("significance", type !== "Forest",
      function () { return { mode: state.significanceMode }; },
      function () {
        state.significanceMode = "none";
        setSelect("pd-significance-mode", "none");
      },
      function (saved) {
        state.significanceMode = saved.mode;
        setSelect("pd-significance-mode", saved.mode);
      });
  }

  function updateControlAvailability() {
    // 1. Declarative type-based visibility via data-plot-types attributes
    applyPlotTypeVisibility();

    // 2. Synchronize internal state + DOM for controls hidden by plot type
    resetStateForHiddenControls();

    // 3. Data-dependent residual logic (supplements type-based visibility)

    // Threshold toggle: disable when no threshold data exists in payload
    var thresholdUnavailable = thresholdReferenceLines.length === 0;
    setControlDisabled("pd-ref-thresholds", thresholdUnavailable);

    // Grouped Mode: also hide when no factor map from backend
    var hasFactorMap = groupFactorMapPayload && typeof groupFactorMapPayload === "object" && Object.keys(groupFactorMapPayload).length > 0;
    var groupingSection = document.getElementById("pd-grouping-section");
    if (groupingSection && !hasFactorMap) {
      groupingSection.classList.add("pd-hidden-by-type");
      state.grouping.enabled = false;
      var grpEnabledEl = document.getElementById("pd-group-enabled");
      if (grpEnabledEl) grpEnabledEl.checked = false;
    }

    // Paired lines: data-dependent (needs usable trajectory data)
    updatePairedLineControlState();

    // Reference note text
    updateReferenceNote();

    // Significance form: availability, fallback and the reason shown to the user
    updateSignificanceAvailability();
  }

  var errorOptionsByCentral = {
    mean: [
      { value: "sd", label: "SD" },
      { value: "sem", label: "SEM" },
      { value: "ci95", label: "95% CI" }
    ],
    median: [
      { value: "iqr", label: "IQR" },
      { value: "range", label: "Range (Min-Max)" }
    ]
  };

  function syncErrorMetricOptions(preferredValue) {
    var select = document.getElementById("pd-error-type");
    if (!select) return;

    var currentCentral = state.centralMeasure === "median" ? "median" : "mean";
    var options = errorOptionsByCentral[currentCentral] || errorOptionsByCentral.mean;
    var preferred = preferredValue || state.errorType;
    var hasPreferred = options.some(function (opt) { return opt.value === preferred; });
    var fallback = options[0].value;
    var resolved = hasPreferred ? preferred : fallback;

    select.innerHTML = "";
    options.forEach(function (opt) {
      var optionNode = document.createElement("option");
      optionNode.value = opt.value;
      optionNode.textContent = opt.label;
      select.appendChild(optionNode);
    });
    select.value = resolved;
    state.errorType = resolved;
  }

  var defaultColors = resolvePalette(DEFAULT_PALETTE_NAME, groupOrder.length);
  groupOrder.forEach(function (group, index) {
    state.colors[group] = defaultColors[index % defaultColors.length];
    state.patterns[group] = "";
    state.symbols[group] = "circle";
    state.groupLabels[group] = group;
  });

  if (hasUsableSubjectTrajectories()) {
  }

  function buildGroupingControls() {
    var enabledEl = document.getElementById("pd-group-enabled");
    var modeEl = document.getElementById("pd-group-mode");
    var editorEl = document.getElementById("pd-group-mapping-editor");
    if (!enabledEl || !modeEl || !editorEl) return;

    enabledEl.checked = state.grouping.enabled;
    modeEl.value = state.grouping.mode;
    document.getElementById("pd-group-mode-row").className = state.grouping.enabled ? "pd-row" : "pd-row is-disabled";
    modeEl.disabled = !state.grouping.enabled;

    if (!state.grouping.enabled) {
      editorEl.style.display = "none";
      return;
    }
    editorEl.style.display = "block";
    editorEl.innerHTML = "";
    
    // Auto-populate map if empty but we have factors from backend
    var backendGroups = Object.keys(groupFactorMapPayload);
    if (Object.keys(state.grouping.map).length === 0 && backendGroups.length > 0) {
      backendGroups.forEach(function(g) {
        state.grouping.map[g] = {
          major: groupFactorMapPayload[g].major,
          minor: groupFactorMapPayload[g].minor
        };
      });
    }

    groupOrder.forEach(function (group) {
      var row = document.createElement("div");
      row.className = "pd-row";
      row.style.marginBottom = "5px";
      
      var label = document.createElement("label");
      label.textContent = group;
      label.style.fontSize = "0.75rem";
      label.style.width = "40%";
      label.style.overflow = "hidden";
      label.style.textOverflow = "ellipsis";
      label.style.whiteSpace = "nowrap";

      var mapObj = state.grouping.map[group] || { major: "", minor: "" };

      var majorInput = document.createElement("input");
      majorInput.type = "text";
      majorInput.value = mapObj.major;
      majorInput.placeholder = "Major (X)";
      majorInput.style.width = "28%";
      majorInput.addEventListener("input", function() {
        state.grouping.map[group] = state.grouping.map[group] || {};
        state.grouping.map[group].major = majorInput.value;
        buildPlot();
      });

      var minorInput = document.createElement("input");
      minorInput.type = "text";
      minorInput.value = mapObj.minor;
      minorInput.placeholder = "Minor (Color)";
      minorInput.style.width = "28%";
      minorInput.addEventListener("input", function() {
        state.grouping.map[group] = state.grouping.map[group] || {};
        state.grouping.map[group].minor = minorInput.value;
        buildPlot();
      });

      row.appendChild(label);
      row.appendChild(majorInput);
      row.appendChild(minorInput);
      editorEl.appendChild(row);
    });
  }

  function setControlDefaults() {
    document.getElementById("pd-plot-type").value = state.plotType;
    document.getElementById("pd-title").value = state.title;
    document.getElementById("pd-x-label").value = state.xLabel;
    document.getElementById("pd-y-label").value = state.yLabel;
    document.getElementById("pd-y-min").value = state.yMin == null ? "" : String(state.yMin);
    document.getElementById("pd-y-max").value = state.yMax == null ? "" : String(state.yMax);
    document.getElementById("pd-font-family").value = state.fontFamily;
    document.getElementById("pd-title-size").value = state.titleSize;
    document.getElementById("pd-axis-size").value = state.axisSize;
    document.getElementById("pd-alpha").value = state.alpha;
    var owDefEl = document.getElementById("pd-outline-width");
    if (owDefEl) owDefEl.value = state.outlineWidth;
    document.getElementById("pd-show-points").checked = state.showPoints;
    
    var pointLayoutEl = document.getElementById("pd-point-layout");
    if (pointLayoutEl) {
      pointLayoutEl.value = state.pointLayout || "jitter";
    }

    document.getElementById("pd-show-error-bars").checked = state.showErrorBars;
    document.getElementById("pd-central-measure").value = state.centralMeasure;
    syncErrorMetricOptions(state.errorType);
    document.getElementById("pd-error-direction").value = state.errorDirection;
    document.getElementById("pd-log-x").checked = state.logX;
    document.getElementById("pd-log-y").checked = state.logY;
    document.getElementById("pd-minor-ticks").checked = state.minorTicks;
    document.getElementById("pd-grid-style").value = state.gridStyle;
    document.getElementById("pd-grid-alpha").value = state.gridAlpha;
    document.getElementById("pd-axis-thickness").value = state.axisThickness;
    document.getElementById("pd-tick-direction").value = state.tickDirection;
    document.getElementById("pd-x-tick-angle").value = state.xTickAngle;
    document.getElementById("pd-y-axis-format").value = state.yAxisFormat;
    document.getElementById("pd-y-min").value = state.yMin == null ? "" : String(state.yMin);
    document.getElementById("pd-y-max").value = state.yMax == null ? "" : String(state.yMax);
    document.getElementById("pd-ref-zero").checked = state.showZeroReferenceLine;
    document.getElementById("pd-ref-unit").checked = state.showUnitReferenceLine;
    document.getElementById("pd-ref-thresholds").checked = state.showThresholdReferenceLines;
    document.getElementById("pd-ref-style").value = state.referenceLineDash;
    document.getElementById("pd-ref-width").value = state.referenceLineWidth;
    document.getElementById("pd-show-legend").checked = state.showLegend;
    document.getElementById("pd-legend-orientation").value = state.legendOrientation;
    document.getElementById("pd-legend-x").value = state.legendX;
    document.getElementById("pd-legend-y").value = state.legendY;
    document.getElementById("pd-legend-xanchor").value = state.legendXAnchor;
    document.getElementById("pd-legend-yanchor").value = state.legendYAnchor;
    setSelect("pd-significance-mode", state.significanceMode);
    document.getElementById("pd-significance-line-width").value = state.significanceLineWidth;
    document.getElementById("pd-significance-spacing").value = state.significanceSpacingScale;
    document.getElementById("pd-significance-size").value = state.significanceStarSize;
    document.getElementById("pd-significance-star-offset").value = state.significanceStarOffset;
    document.getElementById("pd-auto-pattern").checked = state.autoPatternsEnabled;
    document.getElementById("pd-export-width").value = state.exportWidth;
    document.getElementById("pd-export-height").value = state.exportHeight;
    document.getElementById("pd-png-scale").value = String(state.pngScale);
    
    var groupEnabledEl = document.getElementById("pd-group-enabled");
    if (groupEnabledEl) {
      groupEnabledEl.addEventListener("change", function() {
        state.grouping.enabled = groupEnabledEl.checked;
        buildGroupingControls();
        buildPlot();
      });
    }
    
    var groupModeEl = document.getElementById("pd-group-mode");
    if (groupModeEl) {
      groupModeEl.addEventListener("change", function() {
        state.grouping.mode = groupModeEl.value;
        buildPlot();
      });
    }
    
    updatePairedLineControlState();
    buildNodeLabelControls();
    buildGroupingControls();
    updateEncodingControlVisibility();
    updateControlAvailability();
    updateFontPreviewStatus();
  }

  function readStateFromControls() {
    state.plotType = document.getElementById("pd-plot-type").value;
    state.title = document.getElementById("pd-title").value || "";
    state.xLabel = document.getElementById("pd-x-label").value || "";
    state.yLabel = document.getElementById("pd-y-label").value || "";
    state.fontFamily = document.getElementById("pd-font-family").value || "Arial";
    state.titleSize = parseInt(document.getElementById("pd-title-size").value, 10) || 16;
    state.axisSize = parseInt(document.getElementById("pd-axis-size").value, 10) || 12;
    state.alpha = parseFloat(document.getElementById("pd-alpha").value) || 0.85;
    state.showPoints = document.getElementById("pd-show-points").checked;
    
    var pointLayoutEl = document.getElementById("pd-point-layout");
    if (pointLayoutEl) {
      state.pointLayout = pointLayoutEl.value || "jitter";
    }

    if (!hasUsableSubjectTrajectories()) {
    }
    state.showErrorBars = document.getElementById("pd-show-error-bars").checked;
    state.centralMeasure = document.getElementById("pd-central-measure").value || "mean";
    if (["mean", "median"].indexOf(state.centralMeasure) === -1) {
      state.centralMeasure = "mean";
    }
    var owEl = document.getElementById("pd-outline-width");
    if (owEl) {
      var owVal = parseFloat(owEl.value);
      state.outlineWidth = Number.isFinite(owVal) ? Math.min(6, Math.max(0, owVal)) : 2;
    }
    syncErrorMetricOptions(document.getElementById("pd-error-type").value);
    state.errorType = document.getElementById("pd-error-type").value || state.errorType || "sd";
    state.errorDirection = document.getElementById("pd-error-direction").value || "both";
    if (["both", "plus", "minus"].indexOf(state.errorDirection) === -1) {
      state.errorDirection = "both";
    }
    state.logX = document.getElementById("pd-log-x").checked;
    state.logY = document.getElementById("pd-log-y").checked;
    state.minorTicks = document.getElementById("pd-minor-ticks").checked;
    state.gridStyle = document.getElementById("pd-grid-style").value || "none";
    state.gridAlpha = parseFloat(document.getElementById("pd-grid-alpha").value);
    if (!Number.isFinite(state.gridAlpha)) state.gridAlpha = 0.3;
    state.gridAlpha = Math.min(1, Math.max(0.05, state.gridAlpha));
    state.axisThickness = parseFloat(document.getElementById("pd-axis-thickness").value);
    if (!Number.isFinite(state.axisThickness)) state.axisThickness = 0.7;
    state.axisThickness = Math.min(4, Math.max(0.3, state.axisThickness));
    state.tickDirection = document.getElementById("pd-tick-direction").value || "out";
    state.xTickAngle = parseInt(document.getElementById("pd-x-tick-angle").value, 10);
    if (!Number.isFinite(state.xTickAngle)) state.xTickAngle = 0;
    state.xTickAngle = Math.min(90, Math.max(-90, state.xTickAngle));
    state.yAxisFormat = document.getElementById("pd-y-axis-format").value || "auto";
    var yMinRaw = document.getElementById("pd-y-min").value;
    var yMaxRaw = document.getElementById("pd-y-max").value;
    state.yMin = yMinRaw === "" ? null : parseFloat(yMinRaw);
    state.yMax = yMaxRaw === "" ? null : parseFloat(yMaxRaw);
    if (!Number.isFinite(state.yMin)) state.yMin = null;
    if (!Number.isFinite(state.yMax)) state.yMax = null;
    state.showZeroReferenceLine = document.getElementById("pd-ref-zero").checked;
    state.showUnitReferenceLine = document.getElementById("pd-ref-unit").checked;
    state.showThresholdReferenceLines = document.getElementById("pd-ref-thresholds").checked;
    state.referenceLineDash = document.getElementById("pd-ref-style").value || "dash";
    if (["solid", "dash", "dot", "dashdot"].indexOf(state.referenceLineDash) === -1) {
      state.referenceLineDash = "dash";
    }
    state.referenceLineWidth = parseFloat(document.getElementById("pd-ref-width").value);
    if (!Number.isFinite(state.referenceLineWidth)) state.referenceLineWidth = 1.5;
    state.referenceLineWidth = Math.min(4, Math.max(0.6, state.referenceLineWidth));
    state.showLegend = document.getElementById("pd-show-legend").checked;
    state.legendOrientation = document.getElementById("pd-legend-orientation").value || "h";
    state.legendX = parseFloat(document.getElementById("pd-legend-x").value);
    if (!Number.isFinite(state.legendX)) state.legendX = 0;
    state.legendY = parseFloat(document.getElementById("pd-legend-y").value);
    if (!Number.isFinite(state.legendY)) state.legendY = 1.1;
    state.legendXAnchor = document.getElementById("pd-legend-xanchor").value || "left";
    state.legendYAnchor = document.getElementById("pd-legend-yanchor").value || "bottom";
    state.significanceMode = document.getElementById("pd-significance-mode").value || "brackets";
    state.significanceLineWidth = parseFloat(document.getElementById("pd-significance-line-width").value);
    if (!Number.isFinite(state.significanceLineWidth)) state.significanceLineWidth = 1.7;
    state.significanceLineWidth = Math.min(4, Math.max(0.8, state.significanceLineWidth));
    state.significanceSpacingScale = parseFloat(document.getElementById("pd-significance-spacing").value);
    if (!Number.isFinite(state.significanceSpacingScale)) state.significanceSpacingScale = 1.0;
    state.significanceSpacingScale = Math.min(2.2, Math.max(0.7, state.significanceSpacingScale));
    state.significanceStarSize = parseFloat(document.getElementById("pd-significance-size").value);
    if (!Number.isFinite(state.significanceStarSize)) state.significanceStarSize = 14;
    state.significanceStarSize = Math.min(36, Math.max(10, state.significanceStarSize));
    state.significanceStarOffset = parseInt(document.getElementById("pd-significance-star-offset").value, 10);
    if (!Number.isFinite(state.significanceStarOffset)) state.significanceStarOffset = 2;
    state.significanceStarOffset = Math.min(30, Math.max(0, state.significanceStarOffset));
    state.autoPatternsEnabled = document.getElementById("pd-auto-pattern").checked;
    state.exportWidth = parseFloat(document.getElementById("pd-export-width").value) || 8;
    state.exportHeight = parseFloat(document.getElementById("pd-export-height").value) || 6;
    state.pngScale = parseFloat(document.getElementById("pd-png-scale").value) || 3;
    updateFontPreviewStatus();

    Array.from(document.querySelectorAll(".pd-node-label-input")).forEach(function (node) {
      if (node.dataset.group) state.groupLabels[node.dataset.group] = node.value;
    });
    Array.from(document.querySelectorAll(".pd-pattern-select")).forEach(function (node) {
      state.patterns[node.dataset.group] = node.value;
    });
    Array.from(document.querySelectorAll(".pd-symbol-select")).forEach(function (node) {
      state.symbols[node.dataset.group] = node.value;
    });

    if (state.autoPatternsEnabled) {
      applyAutoPatterns();
    }

    state.visiblePairIds = Array.from(document.querySelectorAll(".pd-pair-toggle:checked")).map(function (node) {
      return parseInt(node.value, 10);
    });
  }

  function applyPalette(name) {
    var pal = resolvePalette(name, groupOrder.length);
    state.paletteName = name;
    groupOrder.forEach(function (group, index) {
      state.colors[group] = pal[index % pal.length];
    });
    buildColorControls();
    buildPlot();
  }

  function buildColorControls() {
    var root = document.getElementById("pd-color-controls");
    if (!root) return;
    root.innerHTML = "";
    groupOrder.forEach(function (group) {
      var row = document.createElement("div");
      row.className = "pd-color-item";
      var label = document.createElement("label");
      label.textContent = group;
      var input = document.createElement("input");
      input.type = "color";
      input.value = state.colors[group] || "#0f766e";
      input.dataset.group = group;
      input.addEventListener("input", function () {
        state.colors[group] = input.value;
        buildPlot();
      });
      row.appendChild(label);
      row.appendChild(input);
      root.appendChild(row);
    });
  }

  function moveGroup(index, direction) {
    if (direction === -1 && index > 0) {
      var temp = groupOrder[index - 1];
      groupOrder[index - 1] = groupOrder[index];
      groupOrder[index] = temp;
    } else if (direction === 1 && index < groupOrder.length - 1) {
      var temp = groupOrder[index + 1];
      groupOrder[index + 1] = groupOrder[index];
      groupOrder[index] = temp;
    } else {
      return;
    }
    buildNodeLabelControls();
    buildColorControls();
    buildPatternControls();
    buildSymbolControls();
    buildGroupingControls();
    buildPlot();
  }

  function buildNodeLabelControls() {
    var root = document.getElementById("pd-node-label-controls");
    var groupSection = document.getElementById("pd-node-labels-group");
    if (!root) return;
    if (!groupOrder.length) {
      if (groupSection) groupSection.style.display = "none";
      return;
    }
    if (groupSection) groupSection.style.display = "";
    root.innerHTML = "";
    groupOrder.forEach(function (group, index) {
      var row = document.createElement("div");
      row.className = "pd-row";
      var label = document.createElement("label");
      label.textContent = group;
      label.style.fontSize = "0.78rem";
      label.style.color = "var(--muted)";
      var input = document.createElement("input");
      input.type = "text";
      input.value = state.groupLabels[group] !== undefined ? state.groupLabels[group] : group;
      input.dataset.group = group;
      input.className = "pd-node-label-input";
      input.addEventListener("input", function () {
        state.groupLabels[group] = input.value;
        buildPlot();
      });
      
      var btnContainer = document.createElement("div");
      btnContainer.className = "pd-button-row";
      btnContainer.style.marginLeft = "4px";
      
      var upBtn = document.createElement("button");
      upBtn.type = "button";
      upBtn.innerHTML = "↑";
      upBtn.setAttribute("aria-label", "Move up");
      upBtn.onclick = function() { moveGroup(index, -1); };
      
      var downBtn = document.createElement("button");
      downBtn.type = "button";
      downBtn.innerHTML = "↓";
      downBtn.setAttribute("aria-label", "Move down");
      downBtn.onclick = function() { moveGroup(index, 1); };
      
      btnContainer.appendChild(upBtn);
      btnContainer.appendChild(downBtn);

      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(btnContainer);
      root.appendChild(row);
    });
  }

  function applyAutoPatterns() {
    groupOrder.forEach(function (group, index) {
      state.patterns[group] = defaultPatternCycle[index % defaultPatternCycle.length];
    });
  }

  function buildPatternControls() {
    var root = document.getElementById("pd-pattern-controls");
    if (!root) return;
    root.innerHTML = "";
    groupOrder.forEach(function (group, index) {
      var row = document.createElement("div");
      row.className = "pd-encoding-item";
      var label = document.createElement("label");
      label.textContent = group;
      var select = document.createElement("select");
      select.className = "pd-pattern-select";
      select.dataset.group = group;
      var noneOption = document.createElement("option");
      noneOption.value = "";
      noneOption.textContent = "none";
      select.appendChild(noneOption);
      defaultPatternCycle.forEach(function (shape) {
        var option = document.createElement("option");
        option.value = shape;
        option.textContent = shape;
        select.appendChild(option);
      });
      var fallbackPattern = state.autoPatternsEnabled ? defaultPatternCycle[index % defaultPatternCycle.length] : "";
      select.value = state.patterns[group] || fallbackPattern;
      select.disabled = state.autoPatternsEnabled;
      select.addEventListener("change", function () {
        state.patterns[group] = select.value;
        state.autoPatternsEnabled = false;
        var autoToggle = document.getElementById("pd-auto-pattern");
        if (autoToggle) autoToggle.checked = false;
        buildPatternControls();
        buildPlot();
      });
      row.appendChild(label);
      row.appendChild(select);
      root.appendChild(row);
    });
  }

  function buildSymbolControls() {
    var root = document.getElementById("pd-symbol-controls");
    if (!root) return;
    root.innerHTML = "";
    groupOrder.forEach(function (group, index) {
      var row = document.createElement("div");
      row.className = "pd-encoding-item";
      var label = document.createElement("label");
      label.textContent = group;
      var select = document.createElement("select");
      select.className = "pd-symbol-select";
      select.dataset.group = group;
      defaultSymbolCycle.forEach(function (symbol) {
        var option = document.createElement("option");
        option.value = symbol;
        option.textContent = symbol;
        select.appendChild(option);
      });
      select.value = state.symbols[group] || "circle";
      select.addEventListener("change", function () {
        state.symbols[group] = select.value;
        buildPlot();
      });
      row.appendChild(label);
      row.appendChild(select);
      root.appendChild(row);
    });
  }

  function updateEncodingControlVisibility() {
    var patternRoot = document.getElementById("pd-pattern-controls");
    var symbolRoot = document.getElementById("pd-symbol-controls");
    if (patternRoot) {
      patternRoot.style.display = state.plotType === "Bar" ? "grid" : "none";
    }
    if (symbolRoot) {
      var symbolRelevant = state.showPoints || state.plotType === "Raincloud";
      symbolRoot.style.display = symbolRelevant ? "grid" : "none";
    }
  }

  function getPatternForGroup(group, groupIndex) {
    if (state.autoPatternsEnabled) {
      return defaultPatternCycle[groupIndex % defaultPatternCycle.length];
    }
    return state.patterns[group] || "";
  }

  function getSymbolForGroup(group, groupIndex) {
    return state.symbols[group] || "circle";
  }

  function buildPairControls() {
    var root = document.getElementById("pd-pair-controls");
    if (!root) return;
    root.innerHTML = "";

    var relevant = pairwiseData.filter(function (pair) {
      return pair && pair.group1 && pair.group2 && pair.significant;
    });

    if (!relevant.length) {
      root.textContent = "No significant pairs available.";
      return;
    }

    state.visiblePairIds = relevant.map(function (pair) { return pair.pair_id; });

    relevant.forEach(function (pair) {
      var row = document.createElement("label");
      row.className = "pd-pair-item";
      var txt = document.createElement("span");
      txt.textContent = (pair.group1 + " vs " + pair.group2 + " " + (pair.stars || "")).trim();
      var check = document.createElement("input");
      check.type = "checkbox";
      check.className = "pd-pair-toggle";
      check.value = String(pair.pair_id);
      check.checked = true;
      check.addEventListener("change", buildPlot);
      row.appendChild(txt);
      row.appendChild(check);
      root.appendChild(row);
    });
  }

  function groupIndexMap() {
    var map = {};
    groupOrder.forEach(function (group, index) {
      map[group] = index + 1;
    });
    return map;
  }

  function groupValues(group) {
    return (plotData[group] || []).filter(function (item) {
      return typeof item === "number" && Number.isFinite(item);
    });
  }

  function getStat(group, key) {
    var stats = plotStats[group] || {};
    if (isFiniteNumber(stats[key])) {
      return stats[key];
    }
    return Number.NaN;
  }

  function getBoxSummary(group) {
    var summary = plotStats[group] || {};
    if (
      isFiniteNumber(summary.q1) &&
      isFiniteNumber(summary.median) &&
      isFiniteNumber(summary.q3) &&
      isFiniteNumber(summary.lower_fence) &&
      isFiniteNumber(summary.upper_fence)
    ) {
      return {
        q1: summary.q1,
        median: summary.median,
        q3: summary.q3,
        lowerFence: summary.lower_fence,
        upperFence: summary.upper_fence
      };
    }
    return null;
  }

  function buildPairedLineTraces(idxMap) {
    // Paired-line (spaghetti) overlay was removed from the plot designer.
    return [];
  }

  function getErrorMetricLabel() {
    var currentCentral = state.centralMeasure === "median" ? "median" : "mean";
    var options = errorOptionsByCentral[currentCentral] || [];
    var match = options.find(function (opt) { return opt.value === state.errorType; });
    return match ? match.label : state.errorType.toUpperCase();
  }

  function getBarSummaryAndErrors(group) {
    var center = state.centralMeasure === "median" ? getStat(group, "median") : getStat(group, "mean");
    if (!isFiniteNumber(center)) {
      return null;
    }

    var lowerErr = 0;
    var upperErr = 0;

    if (state.centralMeasure === "mean") {
      if (state.errorType === "sem" || state.errorType === "sd") {
        var spread = getStat(group, state.errorType);
        if (isFiniteNumber(spread)) {
          lowerErr = Math.abs(spread);
          upperErr = Math.abs(spread);
        }
      } else if (state.errorType === "ci95") {
        var ciLower = getStat(group, "ci95_lower");
        var ciUpper = getStat(group, "ci95_upper");
        if (isFiniteNumber(ciLower) && isFiniteNumber(ciUpper)) {
          lowerErr = Math.abs(center - ciLower);
          upperErr = Math.abs(ciUpper - center);
        }
      }
    } else {
      if (state.errorType === "iqr") {
        var q1 = getStat(group, "q1");
        var q3 = getStat(group, "q3");
        if (isFiniteNumber(q1) && isFiniteNumber(q3)) {
          lowerErr = Math.abs(center - q1);
          upperErr = Math.abs(q3 - center);
        }
      } else if (state.errorType === "range") {
        var minValue = getStat(group, "min");
        var maxValue = getStat(group, "max");
        if (isFiniteNumber(minValue) && isFiniteNumber(maxValue)) {
          lowerErr = Math.abs(center - minValue);
          upperErr = Math.abs(maxValue - center);
        }
      }
    }

    return {
      center: center,
      lowerErr: Math.max(0, lowerErr),
      upperErr: Math.max(0, upperErr)
    };
  }

  function buildTraces() {
    var traces = [];
    var idxMap = groupIndexMap();
    var lowerBounds = [];
    var upperBounds = [];

    function getXForShape(group, length) {
      if (state.grouping.enabled) {
        var mapObj = state.grouping.map[group] || { major: "", minor: "" };
        var majors = [];
        var minors = [];
        for (var i = 0; i < length; i++) {
          majors.push(mapObj.major || "");
          minors.push(mapObj.minor || "");
        }
        return [majors, minors];
      } else {
        var numVal = idxMap[group];
        var arr = [];
        for (var i = 0; i < length; i++) arr.push(numVal);
        return arr;
      }
    }

    function getJitterBase(group) {
      return state.grouping.enabled ? (idxMap[group] - 1) : idxMap[group];
    }

    function calculateBeeswarmOffsets(yValues, amplitude) {
      var offsets = new Array(yValues.length).fill(0);
      var items = yValues.map(function(v, i) { return { v: v, i: i }; });
      items.sort(function(a, b) { return a.v - b.v; });
      if (!items.length) return offsets;
      var ySpan = items[items.length - 1].v - items[0].v;
      var binSize = Math.max(ySpan * 0.05, 1e-9);
      var currentBin = [];
      var currentBinY = items[0].v;
      function flushBin() {
        if (!currentBin.length) return;
        var n = currentBin.length;
        var step = (amplitude * 2) / Math.max(1, n);
        var start = -(n - 1) * step / 2;
        currentBin.forEach(function(item, idx) {
          offsets[item.i] = start + idx * step;
        });
        currentBin = [];
      }
      items.forEach(function(item) {
        if (item.v - currentBinY > binSize) {
          flushBin();
          currentBinY = item.v;
        }
        currentBin.push(item);
      });
      flushBin();
      return offsets;
    }

    function getPointXOffsets(group, values, pointOffset, pointJitter, groupIndex) {
      if (state.pointLayout === "beeswarm") {
        var swarms = calculateBeeswarmOffsets(values, pointJitter || 0.22);
        return values.map(function(_, i) { return getJitterBase(group) + (pointOffset || 0) + swarms[i]; });
      }
      return values.map(function(_, pointIndex) {
        return stableJitter(getJitterBase(group) + (pointOffset || 0), pointIndex, groupIndex !== undefined ? groupIndex : (idxMap[group] - 1), pointJitter || 0.22);
      });
    }

    groupOrder.forEach(function (group) {
      var gMin = getStat(group, "min");
      var gMax = getStat(group, "max");
      if (isFiniteNumber(gMin) && isFiniteNumber(gMax)) {
        lowerBounds.push(gMin);
        upperBounds.push(gMax);
      }
    });

    if (!lowerBounds.length || !upperBounds.length) {
      groupOrder.forEach(function (group) {
        var values = groupValues(group);
        if (!values.length) return;
        lowerBounds.push(Math.min.apply(null, values));
        upperBounds.push(Math.max.apply(null, values));
      });
    }

    if (!lowerBounds.length || !upperBounds.length) {
      return { traces: traces, yMin: 0, yMax: 1, idxMap: idxMap };
    }

    if (state.plotType === "Bar") {
      groupOrder.forEach(function (group, groupIndex) {
        var x = idxMap[group];
        var barSummary = getBarSummaryAndErrors(group);
        if (!barSummary) {
          return;
        }
        var centerValue = barSummary.center;
        var lowerErr = barSummary.lowerErr;
        var upperErr = barSummary.upperErr;

        var errorConfig = { visible: false };
        if (state.showErrorBars) {
          var baseError = {
            type: "data",
            visible: true,
            thickness: Math.max(1, state.axisThickness),
            width: state.errorDirection === "both" ? 5 : 0
          };
          if (state.errorDirection === "plus") {
            errorConfig = Object.assign({}, baseError, {
              symmetric: false,
              array: [upperErr],
              arrayminus: [0]
            });
          } else if (state.errorDirection === "minus") {
            errorConfig = Object.assign({}, baseError, {
              symmetric: false,
              array: [0],
              arrayminus: [lowerErr]
            });
          } else {
            errorConfig = Object.assign({}, baseError, {
              symmetric: false,
              array: [upperErr],
              arrayminus: [lowerErr]
            });
          }
        }

        traces.push({
          type: "bar",
          x: getXForShape(group, 1),
          y: [centerValue],
          name: group,
          legendgroup: group,
          marker: {
            color: state.colors[group],
            opacity: state.alpha,
            line: shapeOutline(),
            pattern: {
              shape: getPatternForGroup(group, groupIndex),
              solidity: 0.4,
              size: 9
            }
          },
          error_y: errorConfig,
          showlegend: state.showLegend
        });

        if (!state.showPoints) return;
        var values = groupValues(group);
        if (!values.length) return;
        traces.push({
          type: "scatter",
          mode: "markers",
          x: getPointXOffsets(group, values, 0, 0.22, groupIndex),
          y: values,
          marker: {
            color: plotStyle.pointFillColor,
            symbol: getSymbolForGroup(group, groupIndex),
            size: plotStyle.pointSize,
            opacity: 0.7,
            line: pointEdge()
          },
          legendgroup: group,
          name: group + " points",
          hoverinfo: "x+y",
          showlegend: false
        });
      });
    } else if (state.plotType === "Box") {
      groupOrder.forEach(function (group, groupIndex) {
        var values = groupValues(group);
        var summary = getBoxSummary(group);
        if (summary) {
          traces.push({
            type: "box",
            name: group,
            legendgroup: group,
            x: getXForShape(group, 1),
            q1: [summary.q1],
            median: [summary.median],
            q3: [summary.q3],
            lowerfence: [summary.lowerFence],
            upperfence: [summary.upperFence],
            boxpoints: false,
            marker: { color: state.colors[group], size: 6, opacity: 0.7 },
            line: shapeOutline(),
            fillcolor: state.colors[group],
            opacity: state.alpha,
            showlegend: state.showLegend
          });
        } else if (values.length) {
          traces.push({
            type: "box",
            name: group,
            legendgroup: group,
            x: getXForShape(group, values.length),
            y: values,
            boxpoints: false,
            jitter: 0.3,
            pointpos: 0,
            marker: { color: state.colors[group], size: 6, opacity: 0.7 },
            line: shapeOutline(),
            fillcolor: state.colors[group],
            opacity: state.alpha,
            showlegend: state.showLegend
          });
        }

        if (!state.showPoints || !values.length) return;
        traces.push({
          type: "scatter",
          mode: "markers",
          x: getPointXOffsets(group, values, 0, 0.22, undefined),
          y: values,
          marker: {
            color: plotStyle.pointFillColor,
            symbol: getSymbolForGroup(group, groupIndex),
            size: plotStyle.pointSize,
            opacity: 0.7,
            line: pointEdge()
          },
          legendgroup: group,
          name: group + " points",
          hoverinfo: "x+y",
          showlegend: false
        });
      });
    } else if (state.plotType === "Violin") {
      groupOrder.forEach(function (group, groupIndex) {
        var values = groupValues(group);
        if (!values.length) return;
        traces.push({
          type: "violin",
          name: group,
          legendgroup: group,
          x: getXForShape(group, values.length),
          y: values,
          points: state.showPoints ? "all" : false,
          jitter: 0.28,
          pointpos: 0,
          box: { visible: true },
          meanline: { visible: true },
          marker: {
            color: plotStyle.pointFillColor,
            symbol: getSymbolForGroup(group, groupIndex),
            size: plotStyle.pointSize,
            opacity: 0.65,
            line: pointEdge()
          },
          line: shapeOutline(),
          fillcolor: state.colors[group],
          opacity: state.alpha,
          showlegend: state.showLegend
        });
      });
    } else if (state.plotType === "Raincloud") {
      groupOrder.forEach(function (group, groupIndex) {
        var values = groupValues(group);
        if (!values.length) return;
        var baseX = idxMap[group];
        var pointOffset = -0.18;
        var pointJitter = 0.26;
        traces.push({
          type: "violin",
          name: group + " density",
          legendgroup: group,
          orientation: "h",
          x: values,
          y: getXForShape(group, values.length),
          side: "positive",
          points: false,
          box: { visible: false },
          meanline: { visible: false },
          width: 0.88,
          alignmentgroup: "raincloud-" + group,
          offsetgroup: "raincloud-" + group,
          line: shapeOutline(),
          fillcolor: state.colors[group],
          opacity: Math.min(0.75, state.alpha),
          showlegend: false
        });

        traces.push({
          type: "box",
          orientation: "h",
          name: group,
          legendgroup: group,
          x: values,
          y: values.map(function () { return baseX; }),
          boxpoints: false,
          alignmentgroup: "raincloud-" + group,
          offsetgroup: "raincloud-" + group,
          marker: { color: state.colors[group] },
          line: shapeOutline(),
          fillcolor: "rgba(255,255,255,0.28)",
          width: 0.24,
          opacity: 1,
          showlegend: !state.showPoints && state.showLegend
        });

        if (state.showPoints) {
          traces.push({
            type: "scatter",
            mode: "markers",
            x: values,
            y: getPointXOffsets(group, values, pointOffset, pointJitter, groupIndex),
            marker: {
              // Raincloud is the one plot type whose points take the group's own
              // colour instead of the shared black fill: the cloud, the box and
              // the rain read as one unit that way. Bar/Box/Violin keep black
              // points, where the fill is the group colour and black reads best
              // against it. The black outline from pointEdge() stays either way.
              color: state.colors[group],
              symbol: getSymbolForGroup(group, groupIndex),
              size: plotStyle.pointSize,
              opacity: 0.6,
              line: pointEdge()
            },
            legendgroup: group,
            name: group,
            showlegend: state.showLegend
          });
        }
      });
    } else if (state.plotType === "Forest") {
      var validPairsForForest = pairwiseData.filter(function(p) { return p.effect_size != null; });
      if (validPairsForForest.length === 0) {
        return { traces: [], yMin: 0, yMax: 1, idxMap: idxMap, warning: "Forest plot requires effect size and confidence intervals from post-hoc tests." };
      }
      validPairsForForest.reverse();
      
      var isRatio = validPairsForForest[0].is_ratio === true;
      
      traces.push({
        type: "scatter",
        mode: "markers",
        x: validPairsForForest.map(function(p) { return p.effect_size; }),
        y: validPairsForForest.map(function(p) { return p.comparison; }),
        error_x: {
          type: "data",
          symmetric: false,
          array: validPairsForForest.map(function(p) { return p.ci_upper != null ? Math.max(0, p.ci_upper - p.effect_size) : 0; }),
          arrayminus: validPairsForForest.map(function(p) { return p.ci_lower != null ? Math.max(0, p.effect_size - p.ci_lower) : 0; }),
          visible: true,
          color: "#16313a",
          thickness: 1.5,
          width: 4
        },
        marker: { size: 10, color: "#0f766e", symbol: "square" },
        showlegend: false
      });
      
      var minEffForest = Math.min.apply(null, validPairsForForest.map(function(p) { return p.ci_lower != null ? p.ci_lower : p.effect_size; }));
      var maxEffForest = Math.max.apply(null, validPairsForForest.map(function(p) { return p.ci_upper != null ? p.ci_upper : p.effect_size; }));
      return { traces: traces, yMin: minEffForest, yMax: maxEffForest, idxMap: idxMap, isHorizontalForest: true, isRatioEffect: isRatio };

    } else if (state.plotType === "Estimation") {
      var validPairsForEst = pairwiseData.filter(function(p) { return p.effect_size != null; });
      if (validPairsForEst.length === 0) {
        return { traces: [], yMin: 0, yMax: 1, idxMap: idxMap, warning: "Estimation plot requires effect size and confidence intervals from post-hoc tests." };
      }
      
      var counts = {};
      validPairsForEst.forEach(function(p) {
        counts[p.group1] = (counts[p.group1] || 0) + 1;
        counts[p.group2] = (counts[p.group2] || 0) + 1;
      });
      
      var commonControl = null;
      Object.keys(counts).forEach(function(g) {
         if (counts[g] === validPairsForEst.length) commonControl = g;
      });
      
      if (!commonControl) {
         return { traces: [], yMin: 0, yMax: 1, idxMap: idxMap, warning: "Estimation plot currently requires a control-referenced design (e.g., Dunnett's). All-pairwise (Tukey) is not supported in this view." };
      }

      // Upper Panel: Raw Data Scatter
      groupOrder.forEach(function (group, groupIndex) {
        var values = groupValues(group);
        if (!values.length) return;
        traces.push({
          type: "scatter",
          mode: "markers",
          x: getPointXOffsets(group, values, 0, 0.22, groupIndex),
          y: values,
          marker: {
            color: plotStyle.pointFillColor,
            symbol: getSymbolForGroup(group, groupIndex),
            size: plotStyle.pointSize,
            opacity: 0.7,
            line: pointEdge()
          },
          legendgroup: group,
          name: group,
          showlegend: state.showLegend,
          yaxis: "y"
        });
      });
      
      // Lower Panel: Effect Sizes
      var isRatioEst = validPairsForEst[0].is_ratio === true;
      
      var yMinEff = Infinity;
      var yMaxEff = -Infinity;
      
      validPairsForEst.forEach(function(p) {
         var targetGroup = p.group1 === commonControl ? p.group2 : p.group1;
         traces.push({
           type: "scatter",
           mode: "markers",
           x: getXForShape(targetGroup, 1),
           y: [p.effect_size],
           error_y: {
             type: "data",
             symmetric: false,
             array: [p.ci_upper != null ? Math.max(0, p.ci_upper - p.effect_size) : 0],
             arrayminus: [p.ci_lower != null ? Math.max(0, p.effect_size - p.ci_lower) : 0],
             visible: true,
             color: state.colors[targetGroup] || "#16313a",
             thickness: 2,
             width: 6
           },
           marker: { size: 10, color: state.colors[targetGroup] || "#16313a", symbol: "triangle-up" },
           showlegend: false,
           yaxis: "y2"
         });
         
         if (p.ci_lower != null) yMinEff = Math.min(yMinEff, p.ci_lower);
         else yMinEff = Math.min(yMinEff, p.effect_size);
         
         if (p.ci_upper != null) yMaxEff = Math.max(yMaxEff, p.ci_upper);
         else yMaxEff = Math.max(yMaxEff, p.effect_size);
      });
      
      return { 
        traces: traces, 
        yMin: Math.min.apply(null, lowerBounds), 
        yMax: Math.max.apply(null, upperBounds), 
        idxMap: idxMap, 
        isEstimation: true,
        yMinEff: yMinEff,
        yMaxEff: yMaxEff,
        isRatioEffect: isRatioEst
      };
    }

    return {
      traces: traces,
      yMin: Math.min.apply(null, lowerBounds),
      yMax: Math.max.apply(null, upperBounds),
      idxMap: idxMap
    };
  }

  // Upper edge of each group as drawn, error bars included. Mirrors
  // _group_tops() in report_charts.py. Brackets collapse this to a single
  // baseline; the letter display needs it per group.
  function groupTops() {
    var tops = {};
    groupOrder.forEach(function (group) {
      var top;
      if (state.plotType === "Bar") {
        var barSummary = getBarSummaryAndErrors(group);
        if (!barSummary) return;
        top = barSummary.center;
        if (state.showErrorBars) top += barSummary.upperErr;
      } else {
        top = getStat(group, "max");
      }
      if (isFiniteNumber(top)) tops[group] = top;
    });
    return tops;
  }

  // The violin body is a KDE that overshoots the data maximum, so any
  // annotation placed at the data max collides with the visible tip.
  function violinHeadroom(tops) {
    if (state.plotType !== "Violin") return 0;
    var values = Object.keys(tops).map(function (g) { return tops[g]; });
    if (!values.length) return 0;
    var lower = [];
    groupOrder.forEach(function (group) {
      var mn = getStat(group, "min");
      if (isFiniteNumber(mn)) lower.push(mn);
    });
    var dataMin = lower.length ? Math.min.apply(null, lower) : 0;
    return Math.max((Math.max.apply(null, values) - dataMin) * 0.30, 1.5);
  }

  function bracketYBase() {
    var tops = groupTops();
    var candidates = groupOrder.map(function (g) { return tops[g]; })
      .filter(function (v) { return isFiniteNumber(v); });
    if (!candidates.length) return null;
    var dataMax = Math.max.apply(null, candidates);
    // Violin KDE rendering overshoots the data maximum; add buffer so brackets
    // start above the visible violin tip rather than colliding with it.
    // KDE bandwidth can push the violin tip well beyond the data max, especially
    // when data is clustered near the extremes — use a larger buffer (30 %).
    return dataMax + violinHeadroom(tops);
  }

  function assignLanes(activePairs, yBase, yMin, yMax) {
    var sorted = activePairs.slice().sort(function (a, b) {
      var aSpan = Math.abs(a.i2 - a.i1);
      var bSpan = Math.abs(b.i2 - b.i1);
      if (aSpan !== bSpan) return aSpan - bSpan;
      return a.i1 - b.i1;
    });

    var laneIntervals = [];
    var placedLabels = [];
    var placed = [];

    function laneY(lane) {
      var linearStep = Math.max((Math.abs(yMax - yMin) || Math.abs(yBase) || 1) * 0.1, 0.2) * state.significanceSpacingScale;
      var logStep = 0.08 * state.significanceSpacingScale;
      if (state.logY) {
        return Math.pow(10, Math.log10(yBase) + (lane + 1) * logStep);
      }
      return yBase + (lane + 1) * linearStep;
    }

    function hasIntervalCollision(lane, interval) {
      var used = laneIntervals[lane] || [];
      return used.some(function (entry) {
        return !(interval.end < entry.start || interval.start > entry.end);
      });
    }

    function hasLabelCollision(candidateX, candidateY, stars) {
      var scale = Math.max(0.8, state.significanceStarSize / 14);
      var textLen = Math.max(1, String(stars || "*").length);
      var labelHalfX = Math.max(0.16 * scale, 0.08 * textLen * scale);
      var yGap = state.logY
        ? candidateY * (0.045 * scale)
        : Math.max((Math.abs(yMax - yMin) || 1) * (0.03 * scale), 0.15 * scale);
      var x0 = candidateX - labelHalfX;
      var x1 = candidateX + labelHalfX;
      return placedLabels.some(function (label) {
        var xOverlap = !(x1 < label.x0 || x0 > label.x1);
        var yOverlap = Math.abs(candidateY - label.y) < yGap;
        return xOverlap && yOverlap;
      });
    }

    sorted.forEach(function (pair) {
      var interval = { start: Math.min(pair.i1, pair.i2), end: Math.max(pair.i1, pair.i2) };
      var labelCenterX = (interval.start + interval.end) / 2;
      var laneIndex = 0;
      while (laneIndex < 100) {
        var y = laneY(laneIndex);
        var intervalBlocked = hasIntervalCollision(laneIndex, interval);
        var labelBlocked = hasLabelCollision(labelCenterX, y, pair.stars);
        if (!intervalBlocked && !labelBlocked) {
          break;
        }
        laneIndex += 1;
      }

      if (!laneIntervals[laneIndex]) laneIntervals[laneIndex] = [];
      laneIntervals[laneIndex].push(interval);

      var finalY = laneY(laneIndex);
      var scale = Math.max(0.8, state.significanceStarSize / 14);
      var labelHalfX = Math.max(0.16 * scale, 0.08 * Math.max(1, String(pair.stars || "*").length) * scale);
      placedLabels.push({
        x0: labelCenterX - labelHalfX,
        x1: labelCenterX + labelHalfX,
        y: finalY
      });
      placed.push({ pair: pair, lane: laneIndex, y: finalY });
    });
    return placed;
  }

  // Canonical pair list behind every significance layer (brackets today, the
  // letter display below). Mirrors _pairs_for_plot() in report_charts.py so the
  // static report and this designer can never disagree on which comparisons
  // count. Both significant and non-significant pairs come back: brackets take
  // the significant ones, the letter display needs the full matrix.
  function pairsForPlot(idxMap) {
    return pairwiseData.filter(function (pair) {
      return pair && idxMap[pair.group1] && idxMap[pair.group2];
    }).map(function (pair) {
      return {
        pair_id: pair.pair_id,
        group1: pair.group1,
        group2: pair.group2,
        stars: pair.stars || (pair.significant ? "*" : ""),
        significant: !!pair.significant,
        i1: idxMap[pair.group1],
        i2: idxMap[pair.group2]
      };
    });
  }

  // Brackets additionally honour the per-pair checkboxes: hiding one bracket is
  // pure decluttering, every remaining bracket stays true on its own. The letter
  // display must NOT use this filter -- see buildLetters().
  function visibleSignificantPairs(idxMap) {
    return pairsForPlot(idxMap).filter(function (pair) {
      return pair.significant && state.visiblePairIds.indexOf(pair.pair_id) !== -1;
    });
  }

  // ---- Compact letter display -------------------------------------------
  // Mirror of src/analysis/compact_letters.py. Both sides must agree letter for
  // letter on the same data, so keep the two in step: same completeness gate,
  // same Bron-Kerbosch cliques, same sort_by ordering rule.

  // Letters assert something about EVERY pair on the plot ("same letter = not
  // different"). A pair that was never tested is unknown, not equal -- but the
  // clique algorithm cannot tell the two apart. So letters are only honest when
  // the comparisons cover the complete graph. Structural on purpose: it settles
  // every post-hoc the project has, including paired_custom where the user picks
  // the pairs by hand, and any test added later, without a name list to forget.
  function lettersSupported(pairs) {
    var k = groupOrder.length;
    if (k < 2) return { ok: false, reason: "A letter display needs at least two groups." };
    var known = {};
    groupOrder.forEach(function (g) { known[g] = true; });
    var tested = {};
    (pairs || []).forEach(function (pair) {
      if (!known[pair.group1] || !known[pair.group2] || pair.group1 === pair.group2) return;
      var key = [pair.group1, pair.group2].sort().join("\u0000");
      tested[key] = true;
    });
    var required = k * (k - 1) / 2;
    var have = Object.keys(tested).length;
    if (have < required) {
      return {
        ok: false,
        reason: "Letters require all " + required + " pairwise comparisons between the "
          + k + " groups shown; this post-hoc provides " + have
          + ". Comparisons that were never run cannot be shown as \u2018not different\u2019."
      };
    }
    return { ok: true, reason: "" };
  }

  // Two groups share a letter IFF they are not significantly different --
  // equivalently, they sit in a common maximal clique of the non-significance
  // graph. A star ({group} + its non-different partners) is NOT a substitute:
  // on an intransitive pattern (A~B, B~C, A#C) it collapses A, B, C onto one
  // letter and hides the real A-C difference.
  function compactLetters(pairs, sortBy) {
    var n = groupOrder.length;
    if (n === 0) return {};
    if (n === 1) { var one = {}; one[groupOrder[0]] = "a"; return one; }

    var indexOf = {};
    groupOrder.forEach(function (g, i) { indexOf[g] = i; });

    // notDiff[i][j] true = not significantly different. Untested pairs stay
    // true, which is why lettersSupported() has to gate this.
    var notDiff = [];
    for (var i = 0; i < n; i++) {
      notDiff.push([]);
      for (var j = 0; j < n; j++) notDiff[i].push(true);
    }
    (pairs || []).forEach(function (pair) {
      if (!pair.significant) return;
      var a = indexOf[pair.group1], b = indexOf[pair.group2];
      if (a === undefined || b === undefined) return;
      notDiff[a][b] = notDiff[b][a] = false;
    });

    var adj = [];
    for (var v = 0; v < n; v++) {
      var neighbours = {};
      for (var w = 0; w < n; w++) if (w !== v && notDiff[v][w]) neighbours[w] = true;
      adj.push(neighbours);
    }

    function intersect(setObj, neighbours) {
      var out = {};
      Object.keys(setObj).forEach(function (key) { if (neighbours[key]) out[key] = true; });
      return out;
    }

    var cliques = [];
    (function expand(R, P, X) {
      var pKeys = Object.keys(P), xKeys = Object.keys(X);
      if (!pKeys.length && !xKeys.length) { cliques.push(Object.keys(R).map(Number)); return; }
      pKeys.forEach(function (vKey) {
        var nextR = {};
        Object.keys(R).forEach(function (key) { nextR[key] = true; });
        nextR[vKey] = true;
        expand(nextR, intersect(P, adj[vKey]), intersect(X, adj[vKey]));
        delete P[vKey];
        X[vKey] = true;
      });
    })({}, (function () { var all = {}; for (var q = 0; q < n; q++) all[q] = true; return all; })(), {});

    // Deterministic order so letter 'a' lands on the leading group.
    var rank = {};
    for (var r = 0; r < n; r++) {
      rank[r] = sortBy ? [-(sortBy[groupOrder[r]] || 0), r] : [r, r];
    }
    function rankLess(a, b) {
      return rank[a][0] - rank[b][0] || rank[a][1] - rank[b][1];
    }
    cliques.forEach(function (clique) { clique.sort(rankLess); });
    cliques.sort(function (c1, c2) {
      for (var idx = 0; idx < Math.min(c1.length, c2.length); idx++) {
        var cmp = rankLess(c1[idx], c2[idx]);
        if (cmp) return cmp;
      }
      return c1.length - c2.length;
    });

    var alphabet = "abcdefghijklmnopqrstuvwxyz";
    var letters = {};
    groupOrder.forEach(function (g) { letters[g] = ""; });
    cliques.forEach(function (clique, k) {
      var letter = k < 26 ? alphabet[k]
        : alphabet[Math.floor(k / 26) - 1] + alphabet[k % 26];
      clique.forEach(function (memberIndex) {
        letters[groupOrder[memberIndex]] += letter;
      });
    });
    return letters;
  }

  // Default annotation form, shared with _significance_mode() in report_charts.py.
  // Brackets grow as k(k-1)/2 -- 3 at three groups, 6 at four, 15 at six -- so
  // four groups is where letters start paying for themselves. k >= 4 is the only
  // size condition; two and three groups fall through to brackets on their own
  // rather than through a special case.
  function defaultSignificanceMode(pairs) {
    if (lettersSupported(pairs).ok && groupOrder.length >= 4) return "letters";
    return "brackets";
  }

  // Letters are computed from the FULL comparison matrix, never from
  // state.visiblePairIds. Hiding a bracket is decluttering -- every remaining
  // bracket stays true on its own. Dropping a comparison from a letter display
  // is not: two groups that do differ would merge onto a shared letter, so the
  // plot would state the opposite of the result. The pair checkboxes are
  // disabled in this mode (see updateControlAvailability).
  function buildLetters(yMin, yMax, idxMap) {
    var empty = { shapes: [], annotations: [], yAxisMax: yMax };
    if (state.significanceMode !== "letters") return empty;

    var pairs = pairsForPlot(idxMap);
    var support = lettersSupported(pairs);
    if (!support.ok) {
      return { shapes: [], annotations: [], yAxisMax: yMax, warning: support.reason };
    }

    var tops = groupTops();
    var values = groupOrder.map(function (g) { return tops[g]; })
      .filter(function (v) { return isFiniteNumber(v); });
    if (!values.length) return empty;

    var letters = compactLetters(pairs, tops);
    var headroomBase = violinHeadroom(tops);
    var span = Math.max(Math.abs(Math.max.apply(null, values) - Math.min.apply(null, values)), 1e-9);
    var step = Math.max(span * 0.08, Math.abs(Math.max.apply(null, values)) * 0.04, 1e-9)
      * state.significanceSpacingScale;

    var isHorizontal = state.plotType === "Raincloud";
    var annotations = [];
    var axisMax = Math.max.apply(null, values);
    groupOrder.forEach(function (group) {
      var code = letters[group];
      var top = tops[group];
      if (!code || !isFiniteNumber(top)) return;
      var placed = top + headroomBase + step;
      axisMax = Math.max(axisMax, placed);
      annotations.push({
        x: isHorizontal ? placed : idxMap[group],
        y: isHorizontal ? idxMap[group] : placed,
        text: "<b>" + code + "</b>",
        showarrow: false,
        xref: "x",
        yref: "y",
        xanchor: isHorizontal ? "left" : "center",
        yanchor: isHorizontal ? "middle" : "bottom",
        xshift: isHorizontal ? state.significanceStarOffset : 0,
        yshift: isHorizontal ? 0 : state.significanceStarOffset,
        font: { size: state.significanceStarSize, color: "#16313a" }
      });
    });

    if (isHorizontal) {
      return { shapes: [], annotations: annotations, yAxisMax: yMax, xAxisMax: axisMax };
    }
    return { shapes: [], annotations: annotations, yAxisMax: axisMax };
  }

  // One entry point for the significance layer, whichever form it takes.
  function buildSignificanceLayer(yMin, yMax, idxMap) {
    if (state.significanceMode === "letters") return buildLetters(yMin, yMax, idxMap);
    return buildBrackets(yMin, yMax, idxMap);
  }

  function buildBrackets(yMin, yMax, idxMap) {
    if (state.significanceMode !== "brackets" || !state.visiblePairIds.length) {
      return { shapes: [], annotations: [], yAxisMax: yMax };
    }

    if (state.plotType === "Raincloud") {
      var horizontalPairs = visibleSignificantPairs(idxMap);

      if (!horizontalPairs.length) {
        return { shapes: [], annotations: [], yAxisMax: yMax, xAxisMax: null };
      }

      // Raincloud KDE also overshoots the data maximum on the x-axis — apply
      // the same 30 % range buffer used for vertical Violin plots.
      var dataRangeRaincloud = Math.abs(yMax - yMin) || Math.abs(yMax) || 1;
      var xBase = yMax + Math.max(dataRangeRaincloud * 0.30, 1.5);
      if (!Number.isFinite(xBase)) {
        return { shapes: [], annotations: [], yAxisMax: yMax, xAxisMax: null };
      }

      var stepX = Math.max((Math.abs(yMax - yMin) || Math.abs(xBase) || 1) * 0.1, 0.12) * state.significanceSpacingScale;
      var tickX = Math.max(stepX * 0.22, 0.06);
      var bracketLineWidthHorizontal = Math.min(4, Math.max(0.8, state.significanceLineWidth));
      var shapesHorizontal = [];
      var annotationsHorizontal = [];
      var xAxisMax = xBase;

      horizontalPairs.forEach(function (pair, idx) {
        var laneX = xBase + (idx + 1) * stepX;
        var yLow = Math.min(pair.i1, pair.i2);
        var yHigh = Math.max(pair.i1, pair.i2);
        xAxisMax = Math.max(xAxisMax, laneX + tickX * 2.2);

        shapesHorizontal.push(
          { type: "line", x0: laneX, x1: laneX, y0: yLow, y1: yHigh, xref: "x", yref: "y", line: { color: "rgba(22,49,58,0.65)", width: bracketLineWidthHorizontal } },
          { type: "line", x0: laneX - tickX, x1: laneX, y0: yLow, y1: yLow, xref: "x", yref: "y", line: { color: "rgba(22,49,58,0.65)", width: bracketLineWidthHorizontal } },
          { type: "line", x0: laneX - tickX, x1: laneX, y0: yHigh, y1: yHigh, xref: "x", yref: "y", line: { color: "rgba(22,49,58,0.65)", width: bracketLineWidthHorizontal } }
        );

        annotationsHorizontal.push({
          x: laneX + tickX * 0.35,
          y: (yLow + yHigh) / 2,
          text: "<b>" + pair.stars + "</b>",
          showarrow: false,
          xref: "x",
          yref: "y",
          xanchor: "left",
          yanchor: "middle",
          font: { size: state.significanceStarSize, color: "#16313a" }
        });
      });

      return { shapes: shapesHorizontal, annotations: annotationsHorizontal, yAxisMax: yMax, xAxisMax: xAxisMax };
    }

    var visiblePairs = visibleSignificantPairs(idxMap);

    if (!visiblePairs.length) {
      return { shapes: [], annotations: [], yAxisMax: yMax };
    }

    var yBase = bracketYBase();
    if (yBase === null || !Number.isFinite(yBase)) {
      return { shapes: [], annotations: [], yAxisMax: yMax };
    }

    if (state.logY && yBase <= 0) {
      return { shapes: [], annotations: [], yAxisMax: yMax, warning: "Significance hidden: log scale requires positive values." };
    }

    var laneAssignments = assignLanes(visiblePairs, yBase, yMin, yMax);
    var shapes = [];
    var annotations = [];
    var maxBracketY = yBase;
    var linearStep = Math.max((Math.abs(yMax - yMin) || Math.abs(yBase) || 1) * 0.1, 0.2) * state.significanceSpacingScale;
    var bracketLineWidth = Math.min(4, Math.max(0.8, state.significanceLineWidth));

    laneAssignments.forEach(function (entry) {
      var pair = entry.pair;
      var y = entry.y;
      var tick = state.logY ? y * 0.025 : linearStep * 0.28;
      maxBracketY = Math.max(maxBracketY, y + tick);

      shapes.push(
        { type: "line", x0: pair.i1, x1: pair.i2, y0: y, y1: y, xref: "x", yref: "y", line: { color: "rgba(22,49,58,0.65)", width: bracketLineWidth } },
        { type: "line", x0: pair.i1, x1: pair.i1, y0: y - tick, y1: y, xref: "x", yref: "y", line: { color: "rgba(22,49,58,0.65)", width: bracketLineWidth } },
        { type: "line", x0: pair.i2, x1: pair.i2, y0: y - tick, y1: y, xref: "x", yref: "y", line: { color: "rgba(22,49,58,0.65)", width: bracketLineWidth } }
      );

      annotations.push({
        x: (pair.i1 + pair.i2) / 2,
        y: y,
        text: "<b>" + pair.stars + "</b>",
        showarrow: false,
        xref: "x",
        yref: "y",
        xanchor: "center",
        yanchor: "bottom",
        yshift: state.significanceStarOffset,
        font: { size: state.significanceStarSize, color: "#16313a" }
      });
    });

    return { shapes: shapes, annotations: annotations, yAxisMax: maxBracketY };
  }

  function buildReferenceLinesLayer(yMin, yMax) {
    if (state.plotType === "Raincloud") {
      return {
        shapes: [],
        annotations: [],
        rangeCandidates: [],
        warning: "Reference lines are disabled for Raincloud layout."
      };
    }

    var lines = [];
    var warning = null;

    if (state.showZeroReferenceLine) {
      lines.push({
        value: 0,
        label: "y = 0",
        dash: state.referenceLineDash,
        color: "rgba(22,49,58,0.72)",
        width: state.referenceLineWidth
      });
    }

    if (state.showUnitReferenceLine) {
      lines.push({
        value: 1,
        label: "y = 1",
        dash: state.referenceLineDash,
        color: "rgba(15,118,110,0.84)",
        width: state.referenceLineWidth
      });
    }

    if (state.showThresholdReferenceLines) {
      thresholdReferenceLines.forEach(function (line) {
        lines.push({
          value: line.value,
          label: line.label,
          dash: line.dash || state.referenceLineDash,
          color: line.color || "rgba(159,58,56,0.82)",
          width: Number.isFinite(line.width) ? line.width : state.referenceLineWidth
        });
      });
    }

    if (!lines.length) {
      return {
        shapes: [],
        annotations: [],
        rangeCandidates: [],
        warning: warning
      };
    }

    var visibleLines = lines.filter(function (line) {
      if (state.logY && line.value <= 0) {
        warning = "Reference line(s) <= 0 hidden because log-Y is active.";
        return false;
      }
      return Number.isFinite(line.value);
    });

    if (!visibleLines.length) {
      return {
        shapes: [],
        annotations: [],
        rangeCandidates: [],
        warning: warning
      };
    }

    var shapes = visibleLines.map(function (line) {
      return {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: line.value,
        y1: line.value,
        line: {
          color: line.color,
          width: Math.min(4, Math.max(0.6, Number(line.width) || state.referenceLineWidth)),
          dash: line.dash
        }
      };
    });

    var sorted = visibleLines.slice().sort(function (a, b) { return a.value - b.value; });
    var annotations = [];
    var rangeCandidates = visibleLines.map(function (line) { return line.value; });

    if (state.logY) {
      var minLogGap = 0.055;
      var lastLogY = null;
      sorted.forEach(function (line) {
        var targetLogY = Math.log10(line.value);
        if (lastLogY !== null && targetLogY - lastLogY < minLogGap) {
          targetLogY = lastLogY + minLogGap;
        }
        var adjusted = Math.pow(10, targetLogY);
        lastLogY = targetLogY;
        rangeCandidates.push(adjusted);
        annotations.push({
          x: 1.004,
          y: adjusted,
          xref: "paper",
          yref: "y",
          text: String(line.label),
          showarrow: false,
          xanchor: "left",
          yanchor: "middle",
          align: "left",
          font: { size: Math.max(9, state.axisSize - 1), color: line.color }
        });
      });
    } else {
      var minLinearGap = Math.max((Math.abs(yMax - yMin) || 1) * 0.03, 0.12);
      var lastY = null;
      sorted.forEach(function (line) {
        var adjustedY = line.value;
        if (lastY !== null && adjustedY - lastY < minLinearGap) {
          adjustedY = lastY + minLinearGap;
        }
        lastY = adjustedY;
        rangeCandidates.push(adjustedY);
        annotations.push({
          x: 1.004,
          y: adjustedY,
          xref: "paper",
          yref: "y",
          text: String(line.label),
          showarrow: false,
          xanchor: "left",
          yanchor: "middle",
          align: "left",
          font: { size: Math.max(9, state.axisSize - 1), color: line.color }
        });
      });
    }

    return {
      shapes: shapes,
      annotations: annotations,
      rangeCandidates: rangeCandidates,
      warning: warning
    };
  }

  function buildPlot() {
    readStateFromControls();
    updatePairedLineControlState();
    updateEncodingControlVisibility();
    updateControlAvailability();

    var warningNode = document.getElementById("pd-warning");
    if (warningNode) warningNode.textContent = "";
    var warningMessages = [];

    var built = buildTraces();
    var traces = built.traces;
    if (!traces.length) {
      if (warningNode) warningNode.textContent = "No plottable data found.";
      return;
    }

    var yAxisTitle = state.yLabel;
    if (state.plotType === "Bar" && state.showErrorBars) {
      var centralLabel = state.centralMeasure === "median" ? "Median" : "Mean";
      var metricLabel = getErrorMetricLabel();
      var directionToken = " +/- ";
      if (state.errorDirection === "plus") {
        directionToken = " +";
      } else if (state.errorDirection === "minus") {
        directionToken = " -";
      }
      yAxisTitle += " (" + centralLabel + directionToken + metricLabel + ")";
    }

    var yAxis = {
      title: { text: yAxisTitle, font: { size: state.axisSize } },
      type: state.logY ? "log" : "linear",
      showgrid: state.gridStyle === "major" || state.gridStyle === "both",
      zeroline: !state.logY
    };

    var tickMode = "outside";
    var axisMirror = false;
    if (state.tickDirection === "in") {
      tickMode = "inside";
    } else if (state.tickDirection === "inout") {
      tickMode = "outside";
      axisMirror = "ticks";
    }

    Object.assign(yAxis, axisFrame(tickMode, axisMirror));
    if (state.gridStyle !== "none") {
      yAxis.gridwidth = Math.max(0.5, state.axisThickness * 0.75);
      yAxis.gridcolor = "rgba(22,49,58," + state.gridAlpha + ")";
    }
    if (state.minorTicks) {
      yAxis.minor = {
        ticks: tickMode,
        tickwidth: Math.max(0.5, state.axisThickness * 0.75),
        ticklen: Math.max(3, Math.round(3 + state.axisThickness)),
        showgrid: state.gridStyle === "minor" || state.gridStyle === "both"
      };
      if (state.gridStyle === "minor" || state.gridStyle === "both") {
        yAxis.minor.gridcolor = "rgba(22,49,58," + Math.max(0.05, state.gridAlpha * 0.7) + ")";
        yAxis.minor.gridwidth = Math.max(0.5, state.axisThickness * 0.6);
      }
    }

    if (state.yAxisFormat === "scientific") {
      yAxis.tickformat = ".2e";
    } else if (state.yAxisFormat === "percentage") {
      yAxis.tickformat = ".1%";
    } else if (state.yAxisFormat === "decimal") {
      yAxis.tickformat = ".2f";
    }
    
    if (built.isEstimation) {
      yAxis.domain = [0.35, 1.0];
    }

    var isHorizontalRaincloud = state.plotType === "Raincloud";

    if (!isHorizontalRaincloud && state.yMin != null && state.yMax != null && state.yMax > state.yMin) {
      if (state.logY && state.yMin <= 0) {
        warningMessages.push("Y limits ignored: log scale requires y-min > 0.");
      } else {
        yAxis.range = state.logY ? [Math.log10(state.yMin), Math.log10(state.yMax)] : [state.yMin, state.yMax];
      }
    }

    var bracketLayer = buildSignificanceLayer(built.yMin, built.yMax, built.idxMap);
    if (bracketLayer.warning && warningNode) {
      warningMessages.push(bracketLayer.warning);
    }

    var referenceLayer = buildReferenceLinesLayer(built.yMin, built.yMax);
    if (referenceLayer.warning) {
      warningMessages.push(referenceLayer.warning);
    }

    var combinedCandidates = [built.yMin, built.yMax, bracketLayer.yAxisMax].concat(referenceLayer.rangeCandidates || []);
    combinedCandidates = combinedCandidates.filter(function (value) { return Number.isFinite(value); });

    if (!isHorizontalRaincloud && state.logY && !(state.yMin != null && state.yMax != null && state.yMax > state.yMin && state.yMin > 0)) {
      var positiveCandidates = combinedCandidates.filter(function (value) { return value > 0; });
      if (positiveCandidates.length >= 2) {
        var positiveMin = Math.min.apply(null, positiveCandidates);
        var positiveMax = Math.max.apply(null, positiveCandidates);
        var paddedMin = positiveMin * 0.92;
        var paddedMax = positiveMax * 1.08;
        if (paddedMin > 0 && paddedMax > paddedMin) {
          yAxis.range = [Math.log10(paddedMin), Math.log10(paddedMax)];
          yAxis.autorange = false;
        } else {
          yAxis.range = undefined;
          yAxis.autorange = true;
        }
      } else {
        yAxis.range = undefined;
        yAxis.autorange = true;
      }
    } else if (!isHorizontalRaincloud && !(state.yMin != null && state.yMax != null && state.yMax > state.yMin)) {
      if (state.plotType === "Violin") {
        // The violin body is a KDE that overshoots the data extremes by ~2
        // bandwidths on BOTH ends (Plotly spanmode "soft"). combinedCandidates
        // only carries the raw data min/max (built.yMin/yMax) plus any bracket /
        // reference tops, so the fixed 6-8% pad below clips the violin tips:
        // visibly at the top when no brackets stretch it, and the lower tail is
        // never accounted for at all. The overshoot is data-dependent (it can
        // exceed half the data span for tightly clustered groups), so no
        // constant buffer is safe. Hand framing to Plotly autorange instead --
        // it fits its own rendered violin exactly and still expands to cover the
        // bracket / reference annotations (they carry yref:"y"). Manual y-limits
        // and log-Y are resolved by the branches above.
        yAxis.autorange = true;
        yAxis.range = undefined;
      } else {
        var autoMin = Math.min.apply(null, combinedCandidates);
        var autoMax = Math.max.apply(null, combinedCandidates);
        var autoSpan = Math.max(Math.abs(autoMax - autoMin), 1e-9);
        yAxis.range = [autoMin - autoSpan * 0.08, autoMax + autoSpan * 0.06];
      }
    }

    if (state.logX && groupOrder.length < 2) {
      warningMessages.push("Log X has limited effect with fewer than two groups.");
    }

    var legendOutsideRight = state.showLegend && state.legendOrientation === "v" && state.legendX >= 1;
    var legendBottom = state.showLegend && state.legendOrientation === "h" && state.legendY < 0;
    var hasReferenceAnnotations = Array.isArray(referenceLayer.annotations) && referenceLayer.annotations.length > 0;
    var resolvedFontFamily = resolveFontFamilyStack(state.fontFamily);

      var xAxisConfig = {
        title: { text: state.xLabel, font: { size: state.axisSize } },
        tickangle: state.xTickAngle,
        showgrid: state.gridStyle === "major" || state.gridStyle === "both",
        zeroline: false
      };
      Object.assign(xAxisConfig, axisFrame(tickMode, axisMirror));
      
      if (state.grouping.enabled) {
        xAxisConfig.type = "multicategory";
      } else {
        xAxisConfig.type = state.logX ? "log" : "linear";
        xAxisConfig.tickvals = groupOrder.map(function (_, index) { return index + 1; });
        xAxisConfig.ticktext = groupOrder.map(function (g) { return state.groupLabels[g] !== undefined ? state.groupLabels[g] : g; });
        xAxisConfig.range = state.logX ? [Math.max(0.8, 1 - 0.2), groupOrder.length + 0.6] : [0.4, groupOrder.length + 0.6];
      }

    var layout = {
      template: "plotly_white",
      title: { text: state.title, font: { family: resolvedFontFamily, size: state.titleSize } },
      font: { family: resolvedFontFamily, size: state.axisSize, color: "#16313a" },
      margin: { l: 64, r: Math.max(legendOutsideRight ? 160 : 24, hasReferenceAnnotations ? 130 : 24), t: 58, b: legendBottom ? 120 : 68 },
      xaxis: xAxisConfig,
      yaxis: yAxis,
      showlegend: state.showLegend,
      legend: {
        orientation: state.legendOrientation,
        y: state.legendY,
        x: state.legendX,
        xanchor: state.legendXAnchor,
        yanchor: state.legendYAnchor
      },
      shapes: bracketLayer.shapes.concat(referenceLayer.shapes),
      annotations: bracketLayer.annotations.concat(referenceLayer.annotations),
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#fffdf8",
      hovermode: "closest"
    };

    if (built.isEstimation) {
      layout.yaxis2 = Object.assign({}, yAxis, {
        domain: [0.0, 0.22],
        title: { text: "Effect Size", font: { size: Math.max(10, state.axisSize - 2) } },
        range: undefined,
        autorange: true
      });
      layout.shapes.push({
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y2",
        y0: built.isRatioEffect ? 1 : 0,
        y1: built.isRatioEffect ? 1 : 0,
        line: { color: "rgba(22,49,58,0.72)", width: 1.5, dash: "dash" }
      });
    }

    if (built.isHorizontalForest) {
      layout.yaxis = {
        title: { text: "", font: { size: state.axisSize } },
        type: "category",
        // Forest is drawn horizontally, so the group labels live on the y-axis
        // (same as Raincloud). The label-angle control drives them here too, so
        // rotating labels works on every plot type that has any.
        tickangle: state.xTickAngle,
        showgrid: true,
        zeroline: false
      };
      Object.assign(layout.yaxis, axisFrame(tickMode, axisMirror));
      layout.xaxis = {
        title: { text: "Effect Size", font: { size: state.axisSize } },
        type: built.isRatioEffect ? "log" : "linear",
        showgrid: true,
        zeroline: false
      };
      Object.assign(layout.xaxis, axisFrame(tickMode, axisMirror));
      layout.shapes = [{
        type: "line",
        xref: "x",
        x0: built.isRatioEffect ? 1 : 0,
        x1: built.isRatioEffect ? 1 : 0,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color: "rgba(22,49,58,0.72)", width: 1.5, dash: "dash" }
      }];
      layout.annotations = [];
    }

    if (isHorizontalRaincloud) {
      var horizontalXAxis = {
        title: { text: state.yLabel, font: { size: state.axisSize } },
        type: state.logY ? "log" : "linear",
        showgrid: state.gridStyle === "major" || state.gridStyle === "both",
        zeroline: !state.logY
      };
      Object.assign(horizontalXAxis, axisFrame(tickMode, axisMirror));

      if (!state.logY && state.yMin != null && state.yMax != null && state.yMax > state.yMin) {
        horizontalXAxis.range = [state.yMin, state.yMax];
      }

      var horizontalYAxis = {
        title: { text: state.xLabel, font: { size: state.axisSize } },
        tickangle: state.xTickAngle,
        showgrid: state.gridStyle === "major" || state.gridStyle === "both",
        zeroline: false
      };
      Object.assign(horizontalYAxis, axisFrame(tickMode, axisMirror));
      
      if (state.grouping.enabled) {
        horizontalYAxis.type = "multicategory";
      } else {
        horizontalYAxis.type = state.logX ? "log" : "linear";
        horizontalYAxis.tickvals = groupOrder.map(function (_, index) { return index + 1; });
        horizontalYAxis.ticktext = groupOrder.map(function (g) { return state.groupLabels[g] !== undefined ? state.groupLabels[g] : g; });
        horizontalYAxis.range = state.logX ? [Math.max(0.8, 1 - 0.2), groupOrder.length + 0.6] : [0.4, groupOrder.length + 0.6];
      }

      if (state.yMin != null && state.yMax != null && state.yMax > state.yMin) {
        if (state.logX && state.yMin <= 0) {
          warningMessages.push("Y limits ignored: log scale requires y-min > 0.");
        } else {
          horizontalXAxis.range = state.logX ? [Math.log10(state.yMin), Math.log10(state.yMax)] : [state.yMin, state.yMax];
        }
      } else {
        // Raincloud draws a horizontal one-sided KDE whose density overshoots
        // the data extremes on the value axis by ~2 bandwidths, exactly like the
        // vertical Violin. A fixed 5-12% pad clips those tails, so hand the value
        // axis to Plotly autorange -- it frames its own rendered density and
        // still expands to cover the bracket annotations. Manual limits and log
        // scale are resolved by the branches above.
        horizontalXAxis.autorange = true;
        horizontalXAxis.range = undefined;
      }

      if (state.gridStyle !== "none") {
        horizontalXAxis.gridwidth = Math.max(0.5, state.axisThickness * 0.75);
        horizontalXAxis.gridcolor = "rgba(22,49,58," + state.gridAlpha + ")";
      }
      if (state.minorTicks) {
        horizontalXAxis.minor = {
          ticks: tickMode,
          tickwidth: Math.max(0.5, state.axisThickness * 0.75),
          ticklen: Math.max(3, Math.round(3 + state.axisThickness)),
          showgrid: state.gridStyle === "minor" || state.gridStyle === "both"
        };
        if (state.gridStyle === "minor" || state.gridStyle === "both") {
          horizontalXAxis.minor.gridcolor = "rgba(22,49,58," + Math.max(0.05, state.gridAlpha * 0.7) + ")";
          horizontalXAxis.minor.gridwidth = Math.max(0.5, state.axisThickness * 0.6);
        }
      }

      layout.xaxis = horizontalXAxis;
      layout.yaxis = horizontalYAxis;
    }

    if (state.gridStyle !== "none") {
      layout.xaxis.gridwidth = Math.max(0.5, state.axisThickness * 0.75);
      layout.xaxis.gridcolor = "rgba(22,49,58," + state.gridAlpha + ")";
    }
    // Minor ticks belong on the numeric value axis only, never the categorical
    // group-name axis. The value axis already got them above: yAxis for vertical
    // Bar/Box/Violin, horizontalXAxis for Raincloud. layout.xaxis here is the
    // categorical axis for vertical plots (group names) -- adding minor ticks to
    // it dropped stray ticks between the group labels -- and is the already-set
    // value axis for Raincloud, so no minor block is needed at this point.

    if (warningNode) {
      warningNode.textContent = warningMessages.join(" ");
    }

    // Every axis grows its own margin to fit its tick labels AND its axis title,
    // at any tick angle (0/45/90), on every plot type -- so long x/y-axis titles
    // and long rotated group labels are never clipped by the fixed-size #pd-plot
    // container. A fixed pixel margin can never fit arbitrary-length rotated text,
    // which is why the clipping kept coming back after every hand-tuned margin.
    // Centralised here (after all plot-type branches have built their axes, and
    // covering x/y/x2/y2 alike) so a new plot type or a second axis can never
    // silently miss it.
    Object.keys(layout).forEach(function (axisKey) {
      if (/^[xy]axis\d*$/.test(axisKey) && layout[axisKey] && typeof layout[axisKey] === "object") {
        layout[axisKey].automargin = true;
      }
    });

    Plotly.react("pd-plot", traces, layout, {
      responsive: true,
      displaylogo: false,
      toImageButtonOptions: { format: "png", filename: "biomedstatx_plot", scale: state.pngScale }
    }).then(function () {
      if (typeof window.BioMedStatXTypesetMath === "function") {
        window.BioMedStatXTypesetMath(plotNode);
      }
    });
  }

  function downloadPlot(format) {
    readStateFromControls();
    var widthPx = Math.max(1, Math.round(state.exportWidth * 96));
    var heightPx = Math.max(1, Math.round(state.exportHeight * 96));
    var scale = format === "png" ? state.pngScale : 1;
    Plotly.downloadImage("pd-plot", {
      format: format,
      width: widthPx,
      height: heightPx,
      scale: scale,
      filename: "biomedstatx_plot"
    });
  }

  setControlDefaults();
  styleFontSelectOptions();
  buildColorControls();
  buildPatternControls();
  buildSymbolControls();
  buildPairControls();

  // Resolve the initial significance form from the result itself: an all-pairs
  // post-hoc on four or more groups opens in letters, everything else in
  // brackets. The user sees the right form before touching anything.
  state.significanceMode = defaultSignificanceMode(pairsForPlot(groupIndexMap()));
  setSelect("pd-significance-mode", state.significanceMode);

  Array.from(document.querySelectorAll("#plot-designer-panel input, #plot-designer-panel select")).forEach(function (node) {
    node.addEventListener("change", buildPlot);
    node.addEventListener("input", function () {
      if (node.type === "text" || node.type === "number" || node.type === "range") {
        buildPlot();
      }
    });
  });

  var svgBtn = document.getElementById("pd-download-svg");
  if (svgBtn) {
    svgBtn.addEventListener("click", function () { downloadPlot("svg"); });
  }
  var pngBtn = document.getElementById("pd-download-png");
  if (pngBtn) {
    pngBtn.addEventListener("click", function () { downloadPlot("png"); });
  }

  var autoPatternToggle = document.getElementById("pd-auto-pattern");
  if (autoPatternToggle) {
    autoPatternToggle.addEventListener("change", function () {
      state.autoPatternsEnabled = autoPatternToggle.checked;
      if (state.autoPatternsEnabled) {
        applyAutoPatterns();
      }
      buildPatternControls();
      buildPlot();
    });
  }

  var paletteSelect = document.getElementById("pd-palette");
  if (paletteSelect) {
    paletteSelect.value = state.paletteName || DEFAULT_PALETTE_NAME;
    paletteSelect.addEventListener("change", function () {
      applyPalette(paletteSelect.value);
    });
  }

  function applyLegendPreset(preset) {
    if (preset === "inside-top-right") {
      state.legendOrientation = "v";
      state.legendX = 0.99;
      state.legendY = 0.99;
      state.legendXAnchor = "right";
      state.legendYAnchor = "top";
    } else if (preset === "outside-right") {
      state.legendOrientation = "v";
      state.legendX = 1.02;
      state.legendY = 1.0;
      state.legendXAnchor = "left";
      state.legendYAnchor = "top";
    } else if (preset === "bottom-horizontal") {
      state.legendOrientation = "h";
      state.legendX = 0.5;
      state.legendY = -0.2;
      state.legendXAnchor = "center";
      state.legendYAnchor = "top";
    }

    document.getElementById("pd-legend-orientation").value = state.legendOrientation;
    document.getElementById("pd-legend-x").value = state.legendX;
    document.getElementById("pd-legend-y").value = state.legendY;
    document.getElementById("pd-legend-xanchor").value = state.legendXAnchor;
    document.getElementById("pd-legend-yanchor").value = state.legendYAnchor;
    buildPlot();
  }

  var legendPresetInsideBtn = document.getElementById("pd-legend-preset-inside-top-right");
  if (legendPresetInsideBtn) {
    legendPresetInsideBtn.addEventListener("click", function () {
      applyLegendPreset("inside-top-right");
    });
  }

  var legendPresetOutsideBtn = document.getElementById("pd-legend-preset-outside-right");
  if (legendPresetOutsideBtn) {
    legendPresetOutsideBtn.addEventListener("click", function () {
      applyLegendPreset("outside-right");
    });
  }

  var legendPresetBottomBtn = document.getElementById("pd-legend-preset-bottom-horizontal");
  if (legendPresetBottomBtn) {
    legendPresetBottomBtn.addEventListener("click", function () {
      applyLegendPreset("bottom-horizontal");
    });
  }

  if (typeof plotNode.on === "function") {
    plotNode.on("plotly_afterplot", function () {
      if (typeof window.BioMedStatXTypesetMath === "function") {
        window.BioMedStatXTypesetMath(plotNode);
      }
    });
  }

  buildPlot();
})();

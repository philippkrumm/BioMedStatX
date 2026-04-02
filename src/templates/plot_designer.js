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

  var defaultPalette = [
    "#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed", "#0ea5e9", "#ef4444"
  ];
  // Prioritize combinations that remain separable in dense grayscale exports.
  var defaultPatternCycle = ["x", "\\", "/", "-", "|", "+", "."];
  var defaultSymbolCycle = ["diamond", "square", "circle", "cross", "triangle-up"];
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
    showPoints: true,
    showPairedLines: false,
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
    showSignificance: true,
    significanceLineWidth: 1.7,
    significanceSpacingScale: 1.0,
    significanceStarSize: 14,
    exportWidth: 8,
    exportHeight: 6,
    pngScale: 3,
    colors: {},
    patterns: {},
    symbols: {},
    autoPatternsEnabled: false,
    visiblePairIds: []
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
    var wrapper = document.getElementById("pd-paired-lines-wrap");
    var checkbox = document.getElementById("pd-show-paired-lines");
    if (!wrapper || !checkbox) {
      return;
    }
    var available = hasUsableSubjectTrajectories();
    var raincloudMode = state.plotType === "Raincloud";
    wrapper.style.display = available ? "flex" : "none";
    checkbox.disabled = !available || raincloudMode;
    wrapper.classList.toggle("is-disabled", checkbox.disabled);
    if (!available || raincloudMode) {
      checkbox.checked = false;
      state.showPairedLines = false;
    }
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

  function updateControlAvailability() {
    var isBar = state.plotType === "Bar";
    var isRaincloud = state.plotType === "Raincloud";

    ["pd-show-error-bars", "pd-central-measure", "pd-error-type", "pd-error-direction", "pd-auto-pattern"].forEach(function (id) {
      setControlDisabled(id, !isBar);
    });

    if (!isBar) {
      state.showErrorBars = false;
      var showErrorBarsNode = document.getElementById("pd-show-error-bars");
      if (showErrorBarsNode) showErrorBarsNode.checked = false;
    }

    var thresholdToggle = document.getElementById("pd-ref-thresholds");
    var thresholdUnavailable = thresholdReferenceLines.length === 0;

    ["pd-ref-zero", "pd-ref-unit", "pd-ref-style", "pd-ref-width"].forEach(function (id) {
      setControlDisabled(id, isRaincloud);
    });
    setControlDisabled("pd-ref-thresholds", isRaincloud || thresholdUnavailable);

    if (isRaincloud) {
      state.showZeroReferenceLine = false;
      state.showUnitReferenceLine = false;
      state.showThresholdReferenceLines = false;
      var refZeroNode = document.getElementById("pd-ref-zero");
      var refUnitNode = document.getElementById("pd-ref-unit");
      var refThresholdNode = document.getElementById("pd-ref-thresholds");
      if (refZeroNode) refZeroNode.checked = false;
      if (refUnitNode) refUnitNode.checked = false;
      if (refThresholdNode) refThresholdNode.checked = false;
    }

    updatePairedLineControlState();
    updateReferenceNote();
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

  groupOrder.forEach(function (group, index) {
    state.colors[group] = defaultPalette[index % defaultPalette.length];
    state.patterns[group] = "";
    state.symbols[group] = defaultSymbolCycle[index % defaultSymbolCycle.length];
  });

  function setControlDefaults() {
    document.getElementById("pd-plot-type").value = state.plotType;
    document.getElementById("pd-title").value = state.title;
    document.getElementById("pd-x-label").value = state.xLabel;
    document.getElementById("pd-y-label").value = state.yLabel;
    document.getElementById("pd-font-family").value = state.fontFamily;
    document.getElementById("pd-title-size").value = state.titleSize;
    document.getElementById("pd-axis-size").value = state.axisSize;
    document.getElementById("pd-alpha").value = state.alpha;
    document.getElementById("pd-show-points").checked = state.showPoints;
    document.getElementById("pd-show-paired-lines").checked = state.showPairedLines;
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
    document.getElementById("pd-show-significance").checked = state.showSignificance;
    document.getElementById("pd-significance-line-width").value = state.significanceLineWidth;
    document.getElementById("pd-significance-spacing").value = state.significanceSpacingScale;
    document.getElementById("pd-significance-size").value = state.significanceStarSize;
    document.getElementById("pd-auto-pattern").checked = state.autoPatternsEnabled;
    document.getElementById("pd-export-width").value = state.exportWidth;
    document.getElementById("pd-export-height").value = state.exportHeight;
    document.getElementById("pd-png-scale").value = String(state.pngScale);
    updatePairedLineControlState();
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
    state.showPairedLines = document.getElementById("pd-show-paired-lines").checked;
    if (!hasUsableSubjectTrajectories()) {
      state.showPairedLines = false;
    }
    state.showErrorBars = document.getElementById("pd-show-error-bars").checked;
    state.centralMeasure = document.getElementById("pd-central-measure").value || "mean";
    if (["mean", "median"].indexOf(state.centralMeasure) === -1) {
      state.centralMeasure = "mean";
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
    state.showSignificance = document.getElementById("pd-show-significance").checked;
    state.significanceLineWidth = parseFloat(document.getElementById("pd-significance-line-width").value);
    if (!Number.isFinite(state.significanceLineWidth)) state.significanceLineWidth = 1.7;
    state.significanceLineWidth = Math.min(4, Math.max(0.8, state.significanceLineWidth));
    state.significanceSpacingScale = parseFloat(document.getElementById("pd-significance-spacing").value);
    if (!Number.isFinite(state.significanceSpacingScale)) state.significanceSpacingScale = 1.0;
    state.significanceSpacingScale = Math.min(2.2, Math.max(0.7, state.significanceSpacingScale));
    state.significanceStarSize = parseFloat(document.getElementById("pd-significance-size").value);
    if (!Number.isFinite(state.significanceStarSize)) state.significanceStarSize = 14;
    state.significanceStarSize = Math.min(36, Math.max(10, state.significanceStarSize));
    state.autoPatternsEnabled = document.getElementById("pd-auto-pattern").checked;
    state.exportWidth = parseFloat(document.getElementById("pd-export-width").value) || 8;
    state.exportHeight = parseFloat(document.getElementById("pd-export-height").value) || 6;
    state.pngScale = parseFloat(document.getElementById("pd-png-scale").value) || 3;
    updateFontPreviewStatus();

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
      var fallbackSymbol = defaultSymbolCycle[index % defaultSymbolCycle.length];
      select.value = state.symbols[group] || fallbackSymbol;
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
    return state.symbols[group] || defaultSymbolCycle[groupIndex % defaultSymbolCycle.length];
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
    if (!state.showPairedLines || !hasUsableSubjectTrajectories()) {
      return [];
    }

    var traces = [];
    subjectTrajectories.forEach(function (trajectory) {
      if (!trajectory || !Array.isArray(trajectory.points)) {
        return;
      }
      var points = trajectory.points
        .map(function (point) {
          if (!point) return null;
          var group = String(point.group || "");
          var xValue = idxMap[group];
          var yValue = Number(point.value);
          if (!xValue || !Number.isFinite(yValue)) {
            return null;
          }
          return { x: xValue, y: yValue };
        })
        .filter(Boolean)
        .sort(function (a, b) { return a.x - b.x; });

      if (points.length < 2) {
        return;
      }

      traces.push({
        type: "scatter",
        mode: "lines+markers",
        x: points.map(function (p) { return p.x; }),
        y: points.map(function (p) { return p.y; }),
        connectgaps: false,
        line: {
          color: "rgba(22,49,58,0.32)",
          width: 1.1
        },
        marker: {
          color: "rgba(22,49,58,0.45)",
          size: 4,
          symbol: "circle"
        },
        hovertemplate: "Subject: " + String(trajectory.subject_id || "") + "<br>x=%{x}<br>y=%{y:.4g}<extra></extra>",
        showlegend: false,
        name: "Subject trajectory"
      });
    });

    return traces;
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
          x: [x],
          y: [centerValue],
          name: group,
          legendgroup: group,
          marker: {
            color: state.colors[group],
            opacity: state.alpha,
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
          x: values.map(function (_, pointIndex) {
            return stableJitter(idxMap[group], pointIndex, groupIndex, 0.22);
          }),
          y: values,
          marker: {
            color: state.colors[group],
            symbol: getSymbolForGroup(group, groupIndex),
            size: 6,
            opacity: 0.7,
            line: { width: 0.5, color: "#16313a" }
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
            x: [idxMap[group]],
            q1: [summary.q1],
            median: [summary.median],
            q3: [summary.q3],
            lowerfence: [summary.lowerFence],
            upperfence: [summary.upperFence],
            boxpoints: false,
            marker: { color: state.colors[group], size: 6, opacity: 0.7 },
            line: { color: state.colors[group] },
            fillcolor: state.colors[group],
            opacity: state.alpha,
            showlegend: state.showLegend
          });
        } else if (values.length) {
          traces.push({
            type: "box",
            name: group,
            legendgroup: group,
            x: values.map(function () { return idxMap[group]; }),
            y: values,
            boxpoints: false,
            jitter: 0.3,
            pointpos: 0,
            marker: { color: state.colors[group], size: 6, opacity: 0.7 },
            line: { color: state.colors[group] },
            fillcolor: state.colors[group],
            opacity: state.alpha,
            showlegend: state.showLegend
          });
        }

        if (!state.showPoints || !values.length) return;
        traces.push({
          type: "scatter",
          mode: "markers",
          x: values.map(function (_, pointIndex) {
            return stableJitter(idxMap[group], pointIndex, idxMap[group] - 1, 0.22);
          }),
          y: values,
          marker: {
            color: state.colors[group],
            symbol: getSymbolForGroup(group, groupIndex),
            size: 6,
            opacity: 0.7,
            line: { width: 0.5, color: "#16313a" }
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
          x: values.map(function () { return idxMap[group]; }),
          y: values,
          points: state.showPoints ? "all" : false,
          jitter: 0.28,
          pointpos: 0,
          box: { visible: true },
          meanline: { visible: true },
          marker: {
            color: state.colors[group],
            symbol: getSymbolForGroup(group, groupIndex),
            size: 5,
            opacity: 0.65
          },
          line: { color: state.colors[group] },
          fillcolor: state.colors[group],
          opacity: state.alpha,
          showlegend: state.showLegend
        });
      });
    } else {
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
          y: values.map(function () { return baseX; }),
          side: "positive",
          points: false,
          box: { visible: false },
          meanline: { visible: false },
          width: 0.88,
          alignmentgroup: "raincloud-" + group,
          offsetgroup: "raincloud-" + group,
          line: { color: state.colors[group] },
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
          line: { color: "rgba(22,49,58,0.85)", width: 1.2 },
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
            y: values.map(function (_, pointIndex) {
              return stableJitter(baseX + pointOffset, pointIndex, groupIndex, pointJitter);
            }),
            marker: {
              color: state.colors[group],
              symbol: getSymbolForGroup(group, groupIndex),
              size: 5,
              opacity: 0.6,
              line: { width: 0.4, color: "#16313a" }
            },
            legendgroup: group,
            name: group,
            showlegend: state.showLegend
          });
        }
      });
    }

    if (state.showPairedLines && state.plotType !== "Raincloud") {
      traces = traces.concat(buildPairedLineTraces(idxMap));
    }

    return {
      traces: traces,
      yMin: Math.min.apply(null, lowerBounds),
      yMax: Math.max.apply(null, upperBounds),
      idxMap: idxMap
    };
  }

  function bracketYBase() {
    var candidates = [];
    if (state.plotType === "Bar") {
      groupOrder.forEach(function (group) {
        var barSummary = getBarSummaryAndErrors(group);
        if (!barSummary) {
          return;
        }
        var top = barSummary.center;
        if (state.showErrorBars) {
          top += barSummary.upperErr;
        }
        candidates.push(top);
      });
    } else {
      groupOrder.forEach(function (group) {
        var top = getStat(group, "max");
        if (isFiniteNumber(top)) {
          candidates.push(top);
        }
      });
    }
    if (!candidates.length) return null;
    return Math.max.apply(null, candidates);
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

  function buildBrackets(yMin, yMax, idxMap) {
    if (!state.showSignificance || !state.visiblePairIds.length) {
      return { shapes: [], annotations: [], yAxisMax: yMax };
    }

    if (state.plotType === "Raincloud") {
      var horizontalPairs = pairwiseData.filter(function (pair) {
        if (!pair || !pair.significant) return false;
        if (state.visiblePairIds.indexOf(pair.pair_id) === -1) return false;
        return idxMap[pair.group1] && idxMap[pair.group2];
      }).map(function (pair) {
        return {
          stars: pair.stars || "*",
          i1: idxMap[pair.group1],
          i2: idxMap[pair.group2]
        };
      });

      if (!horizontalPairs.length) {
        return { shapes: [], annotations: [], yAxisMax: yMax, xAxisMax: null };
      }

      var xBase = yMax;
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

    var visiblePairs = pairwiseData.filter(function (pair) {
      if (!pair || !pair.significant) return false;
      if (state.visiblePairIds.indexOf(pair.pair_id) === -1) return false;
      return idxMap[pair.group1] && idxMap[pair.group2];
    }).map(function (pair) {
      return {
        group1: pair.group1,
        group2: pair.group2,
        stars: pair.stars || "*",
        i1: idxMap[pair.group1],
        i2: idxMap[pair.group2]
      };
    });

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
        yshift: Math.max(8, Math.round(state.significanceStarSize * 0.55)),
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

    yAxis.showline = true;
    yAxis.linecolor = "rgba(22,49,58,0.75)";
    yAxis.linewidth = Math.max(0.5, state.axisThickness);
    yAxis.ticks = tickMode;
    yAxis.tickwidth = Math.max(0.5, state.axisThickness);
    yAxis.ticklen = Math.max(4, Math.round(4 + state.axisThickness * 2));
    yAxis.mirror = axisMirror;
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

    var isHorizontalRaincloud = state.plotType === "Raincloud";

    if (!isHorizontalRaincloud && state.yMin != null && state.yMax != null && state.yMax > state.yMin) {
      if (state.logY && state.yMin <= 0) {
        warningMessages.push("Y limits ignored: log scale requires y-min > 0.");
      } else {
        yAxis.range = state.logY ? [Math.log10(state.yMin), Math.log10(state.yMax)] : [state.yMin, state.yMax];
      }
    }

    var bracketLayer = buildBrackets(built.yMin, built.yMax, built.idxMap);
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
      var autoMin = Math.min.apply(null, combinedCandidates);
      var autoMax = Math.max.apply(null, combinedCandidates);
      var autoSpan = Math.max(Math.abs(autoMax - autoMin), 1e-9);
      yAxis.range = [autoMin - autoSpan * 0.08, autoMax + autoSpan * 0.06];
    }

    if (state.logX && groupOrder.length < 2) {
      warningMessages.push("Log X has limited effect with fewer than two groups.");
    }

    var legendOutsideRight = state.showLegend && state.legendOrientation === "v" && state.legendX >= 1;
    var legendBottom = state.showLegend && state.legendOrientation === "h" && state.legendY < 0;
    var hasReferenceAnnotations = Array.isArray(referenceLayer.annotations) && referenceLayer.annotations.length > 0;
    var resolvedFontFamily = resolveFontFamilyStack(state.fontFamily);

    var layout = {
      template: "plotly_white",
      title: { text: state.title, font: { family: resolvedFontFamily, size: state.titleSize } },
      font: { family: resolvedFontFamily, size: state.axisSize, color: "#16313a" },
      margin: { l: 64, r: Math.max(legendOutsideRight ? 160 : 24, hasReferenceAnnotations ? 130 : 24), t: 58, b: legendBottom ? 120 : 68 },
      xaxis: {
        title: { text: state.xLabel, font: { size: state.axisSize } },
        tickvals: groupOrder.map(function (_, index) { return index + 1; }),
        ticktext: groupOrder,
        tickangle: state.xTickAngle,
        type: state.logX ? "log" : "linear",
        showgrid: state.gridStyle === "major" || state.gridStyle === "both",
        zeroline: false,
        showline: true,
        linecolor: "rgba(22,49,58,0.75)",
        linewidth: Math.max(0.5, state.axisThickness),
        ticks: tickMode,
        tickwidth: Math.max(0.5, state.axisThickness),
        ticklen: Math.max(4, Math.round(4 + state.axisThickness * 2)),
        mirror: axisMirror,
        range: state.logX ? [Math.max(0.8, 1 - 0.2), groupOrder.length + 0.6] : [0.4, groupOrder.length + 0.6]
      },
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

    if (isHorizontalRaincloud) {
      var horizontalXAxis = {
        title: { text: state.yLabel, font: { size: state.axisSize } },
        type: state.logX ? "log" : "linear",
        showgrid: state.gridStyle === "major" || state.gridStyle === "both",
        zeroline: !state.logX,
        showline: true,
        linecolor: "rgba(22,49,58,0.75)",
        linewidth: Math.max(0.5, state.axisThickness),
        ticks: tickMode,
        tickwidth: Math.max(0.5, state.axisThickness),
        ticklen: Math.max(4, Math.round(4 + state.axisThickness * 2)),
        mirror: axisMirror
      };

      var horizontalYAxis = {
        title: { text: state.xLabel, font: { size: state.axisSize } },
        tickvals: groupOrder.map(function (_, index) { return index + 1; }),
        ticktext: groupOrder,
        showgrid: false,
        zeroline: false,
        showline: true,
        linecolor: "rgba(22,49,58,0.75)",
        linewidth: Math.max(0.5, state.axisThickness),
        ticks: tickMode,
        tickwidth: Math.max(0.5, state.axisThickness),
        ticklen: Math.max(4, Math.round(4 + state.axisThickness * 2)),
        mirror: axisMirror,
        range: [0.5, groupOrder.length + 0.5]
      };

      var xCandidatesHorizontal = [built.yMin, built.yMax];
      if (Number.isFinite(bracketLayer.xAxisMax)) {
        xCandidatesHorizontal.push(bracketLayer.xAxisMax);
      }
      xCandidatesHorizontal = xCandidatesHorizontal.filter(function (value) { return Number.isFinite(value); });

      if (state.yMin != null && state.yMax != null && state.yMax > state.yMin) {
        if (state.logX && state.yMin <= 0) {
          warningMessages.push("Y limits ignored: log scale requires y-min > 0.");
        } else {
          horizontalXAxis.range = state.logX ? [Math.log10(state.yMin), Math.log10(state.yMax)] : [state.yMin, state.yMax];
        }
      } else if (xCandidatesHorizontal.length >= 2) {
        var autoMinH = Math.min.apply(null, xCandidatesHorizontal);
        var autoMaxH = Math.max.apply(null, xCandidatesHorizontal);
        var autoSpanH = Math.max(Math.abs(autoMaxH - autoMinH), 1e-9);
        horizontalXAxis.range = [autoMinH - autoSpanH * 0.05, autoMaxH + autoSpanH * 0.12];
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
    if (state.minorTicks) {
      layout.xaxis.minor = {
        ticks: tickMode,
        tickwidth: Math.max(0.5, state.axisThickness * 0.75),
        ticklen: Math.max(3, Math.round(3 + state.axisThickness)),
        showgrid: state.gridStyle === "minor" || state.gridStyle === "both"
      };
      if (state.gridStyle === "minor" || state.gridStyle === "both") {
        layout.xaxis.minor.gridcolor = "rgba(22,49,58," + Math.max(0.05, state.gridAlpha * 0.7) + ")";
        layout.xaxis.minor.gridwidth = Math.max(0.5, state.axisThickness * 0.6);
      }
    }

    if (warningNode) {
      warningNode.textContent = warningMessages.join(" ");
    }

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

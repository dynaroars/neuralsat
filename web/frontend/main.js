import './style.css';

// Override fetch to automatically append ngrok skip warning headers
const originalFetch = window.fetch;
window.fetch = function (input, init) {
  let url = '';
  if (typeof input === 'string') {
    url = input;
  } else if (input && typeof input === 'object' && input.url) {
    url = input.url;
  }
  
  if (url.includes('ngrok-free.dev')) {
    init = init || {};
    init.headers = init.headers || {};
    if (init.headers instanceof Headers) {
      init.headers.set('ngrok-skip-browser-warning', '69420');
    } else if (Array.isArray(init.headers)) {
      init.headers.push(['ngrok-skip-browser-warning', '69420']);
    } else {
      init.headers['ngrok-skip-browser-warning'] = '69420';
    }
  }
  return originalFetch(input, init);
};

const API_BASE = 'https://oarless-chafflike-chung.ngrok-free.dev/api';
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

const POLL_INITIAL = 1000;
const POLL_MAX = 5000;

let onnxFile = null;
let vnnlibFile = null;
let currentJobId = null;
let pollTimer = null;
let logPollTimer = null;
let pollInterval = POLL_INITIAL;
let startTime = null;
let elapsedTimer = null;
let topologyAnimId = null;

const $ = (id) => document.getElementById(id);

const onnxZone = $('onnx-zone');
const onnxInput = $('onnx-input');
const onnxFileInfo = $('onnx-file-info');
const onnxFilename = $('onnx-filename');
const onnxFilesize = $('onnx-filesize');
const onnxRemove = $('onnx-remove');

const vnnlibZone = $('vnnlib-zone');
const vnnlibInput = $('vnnlib-input');
const vnnlibFileInfo = $('vnnlib-file-info');
const vnnlibFilename = $('vnnlib-filename');
const vnnlibFilesize = $('vnnlib-filesize');
const vnnlibRemove = $('vnnlib-remove');

const verifyBtn = $('verify-btn');
const cancelBtn = $('cancel-btn');
const timeoutInput = $('timeout-input');
const deviceSelect = $('device-select');

const stateIdle = $('state-idle');
const stateRunning = $('state-running');
const stateCompleted = $('state-completed');
const stateError = $('state-error');

const runningMsg = $('running-msg');
const runningElapsed = $('running-elapsed');
const progressFill = $('progress-fill');
const resultBadge = $('result-badge');
const resultDetails = $('result-details');
const errorMsg = $('error-msg');

const terminalContainer = $('terminal-container');
const terminalLog = $('terminal-log');
const terminalScrollAuto = $('terminal-scroll-auto');
const visualizerPanel = $('visualizer-panel');
const topologyContainer = $('topology-container');
const boundsContainer = $('bounds-container');

const examplesGrid = $('examples-grid');

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(0);
  return `${m}m ${s}s`;
}

function transitionDOM(callback) {
  if (document.startViewTransition) {
    document.startViewTransition(callback);
  } else {
    callback();
  }
}

function setupUploadZone(zone, input, fileInfoEl, filenameEl, filesizeEl, removeBtn, accept, onFile) {
  // Click to browse
  zone.addEventListener('click', (e) => {
    if (e.target === removeBtn || removeBtn.contains(e.target)) return;
    input.click();
  });

  // Keyboard
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input.click();
    }
  });

  // File input change
  input.addEventListener('change', () => {
    if (input.files.length > 0) {
      handleFile(input.files[0], zone, fileInfoEl, filenameEl, filesizeEl, accept, onFile);
    }
  });

  // Drag events
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0], zone, fileInfoEl, filenameEl, filesizeEl, accept, onFile);
    }
  });

  // Remove button
  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile(zone, input, fileInfoEl, onFile);
  });
}

function handleFile(file, zone, fileInfoEl, filenameEl, filesizeEl, accept, onFile) {
  // Validate extension
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (ext !== accept) {
    showToast(`Please select a ${accept} file`, 'error');
    return;
  }

  // Validate size
  if (file.size > MAX_UPLOAD_BYTES) {
    showToast(`File too large. Maximum is ${formatBytes(MAX_UPLOAD_BYTES)}`, 'error');
    return;
  }

  // Show file info
  zone.classList.add('has-file');
  zone.querySelector('.upload-zone__content').hidden = true;
  fileInfoEl.hidden = false;
  filenameEl.textContent = file.name;
  filesizeEl.textContent = formatBytes(file.size);

  onFile(file);
  updateVerifyButton();
}

function clearFile(zone, input, fileInfoEl, onFile) {
  zone.classList.remove('has-file');
  zone.querySelector('.upload-zone__content').hidden = false;
  fileInfoEl.hidden = true;
  input.value = '';
  onFile(null);
  updateVerifyButton();
}

function updateVerifyButton() {
  verifyBtn.disabled = !(onnxFile && vnnlibFile) || currentJobId !== null;
}

// upload zones
setupUploadZone(onnxZone, onnxInput, onnxFileInfo, onnxFilename, onnxFilesize, onnxRemove, '.onnx', (f) => { onnxFile = f; });
setupUploadZone(vnnlibZone, vnnlibInput, vnnlibFileInfo, vnnlibFilename, vnnlibFilesize, vnnlibRemove, '.vnnlib', (f) => { vnnlibFile = f; });

function setVerificationRunningUI() {
  transitionDOM(() => {
    showState('running');
    runningMsg.textContent = 'Submitting job…';
    runningElapsed.textContent = '';
    progressFill.style.transform = 'scaleX(0)';
    verifyBtn.disabled = true;
    terminalContainer.hidden = false;
    terminalLog.textContent = 'Uploading files and initializing verify request...\n';
  });
}

async function submitVerificationRequest(formData) {
  const res = await fetch(`${API_BASE}/verify`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Server error' }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}

function handleVerificationError(err) {
  transitionDOM(() => {
    showState('error');
    errorMsg.textContent = err.message;
    currentJobId = null;
    terminalContainer.hidden = true;
    updateVerifyButton();
  });
}

function getVerificationFormData() {
  const timeout = parseInt(timeoutInput.value) || 300;
  const formData = new FormData();
  formData.append('net', onnxFile);
  formData.append('spec', vnnlibFile);
  formData.append('timeout', timeout);
  formData.append('device', deviceSelect.value);
  return { formData, timeout };
}

async function startVerification() {
  if (!onnxFile) return;
  if (!vnnlibFile) return;

  const { formData, timeout } = getVerificationFormData();
  setVerificationRunningUI();

  try {
    const data = await submitVerificationRequest(formData);
    currentJobId = data.job_id;
    startTime = Date.now();

    runningMsg.textContent = 'Verifying…';
    startElapsedTimer(timeout);

    // Render ONNX topology and VNNLib bounds
    renderModelAnalysis(data);

    startPolling();
    startLogPolling();

  } catch (err) {
    handleVerificationError(err);
  }
}

verifyBtn.addEventListener('click', startVerification);

// Cancel Verification Handler
cancelBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  try {
    showToast('Cancelling verification…', 'info');
    const res = await fetch(`${API_BASE}/cancel/${currentJobId}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    showToast('Verification cancelled successfully.', 'success');
    stopTimers();
    currentJobId = null;
    transitionDOM(() => {
      showState('idle');
      terminalContainer.hidden = true;
      updateVerifyButton();
    });
  } catch (err) {
    showToast(`Cancellation failed: ${err.message}`, 'error');
  }
});

function startPolling() {
  pollInterval = POLL_INITIAL;
  pollNext();
}

function pollNext() {
  if (!currentJobId) return;
  pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`${API_BASE}/status/${currentJobId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      handleStatusUpdate(data);
    } catch (err) {
      console.error('Poll error:', err);
      // Keep polling on network errors
      pollInterval = Math.min(pollInterval * 1.5, POLL_MAX);
      pollNext();
    }
  }, pollInterval);
}

const SOLVER_PHASES = [
  'Initializing DPLL(T) solver engine…',
  'Constructing layer interval bounds…',
  'Propagating activation constraints…',
  'Analyzing ReLU states (stable vs unstable)…',
  'Solving linear relaxations (LP solver)…',
  'Splitting nodes in search tree…',
  'Pruning unsatisfiable branches…',
  'Searching for counterexample inputs…'
];

function startElapsedTimer(timeout) {
  let phaseIdx = 0;
  let tickCount = 0;
  runningMsg.textContent = SOLVER_PHASES[0];

  elapsedTimer = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    runningElapsed.textContent = `Elapsed: ${formatTime(elapsed)}`;
    const pct = Math.min((elapsed / timeout) * 100, 98);
    progressFill.style.transform = `scaleX(${pct / 100})`;

    tickCount++;
    if (tickCount % 6 === 0) {
      phaseIdx = (phaseIdx + 1) % SOLVER_PHASES.length;
      runningMsg.textContent = SOLVER_PHASES[phaseIdx];
    }
  }, 500);
}

function showErrorState(errorMsgText) {
  showState('error');
  errorMsg.textContent = errorMsgText || 'An unknown error occurred.';
}

function finalizeJob(data) {
  const finishedJobId = currentJobId;
  stopTimers();
  currentJobId = null;
  updateVerifyButton();

  transitionDOM(() => {
    switch (data.status) {
      case 'completed':
        showResult(data);
        pollLogsFinal(finishedJobId);
        break;
      case 'error':
        showErrorState(data.error);
        pollLogsFinal(finishedJobId);
        break;
      case 'cancelled':
        showState('idle');
        terminalContainer.hidden = true;
        break;
    }
  });
}

function handleStatusUpdate(data) {
  if (data.status === 'queued') {
    runningMsg.textContent = 'Queued — waiting for previous job to finish…';
    pollNext();
    return;
  }

  if (data.status === 'running') {
    runningMsg.textContent = 'Verifying…';
    pollInterval = Math.min(pollInterval * 1.2, POLL_MAX);
    pollNext();
    return;
  }

  finalizeJob(data);
}

function startLogPolling() {
  if (logPollTimer) clearTimeout(logPollTimer);
  pollLogs();
}

function maybeScrollTerminal() {
  if (terminalScrollAuto.checked) {
    const terminalBody = $('terminal-body');
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }
}

async function fetchAndRenderLogs(jobId) {
  const res = await fetch(`${API_BASE}/logs/${jobId}`);
  if (!res.ok) return;
  const data = await res.json();
  if (data.logs === undefined) return;

  terminalLog.textContent = data.logs || 'Initial logs pending...';
  maybeScrollTerminal();
}

async function pollLogs() {
  if (!currentJobId) return;

  try {
    await fetchAndRenderLogs(currentJobId);
  } catch (err) {
    console.error('Log poll error:', err);
  }

  if (currentJobId) {
    logPollTimer = setTimeout(pollLogs, 1000);
  }
}

async function pollLogsFinal(jobId) {
  try {
    await fetchAndRenderLogs(jobId);
  } catch (err) {
    console.error('Log poll error:', err);
  }
}

function renderModelAnalysis(data) {
  const onnxInfo = data.onnx_info;
  const vnnlibInfo = data.vnnlib_info;

  if (!onnxInfo && !vnnlibInfo) {
    transitionDOM(() => {
      visualizerPanel.hidden = true;
    });
    return;
  }

  transitionDOM(() => {
    visualizerPanel.hidden = false;
    renderOnnxTopology(onnxInfo);
    renderVnnlibBounds(vnnlibInfo);
  });
}

function getDimension(shapeArray) {
  const lastDim = (shapeArray || []).pop();
  const val = parseInt(lastDim) || 3;
  return val <= 0 ? 3 : val;
}

function getStyleToken(styles, name, fallback) {
  return styles.getPropertyValue(name).trim() || fallback;
}

function getInputColumn(onnxInfo, accent3) {
  const input = (onnxInfo.inputs || [])[0];
  if (!input) {
    return { type: 'INPUT', name: 'X', shape: '[]', color: accent3, nodeCount: 3 };
  }
  const inputDim = getDimension(input.shape);
  return {
    type: 'INPUT',
    name: input.name,
    shape: `[${input.shape.join(', ')}]`,
    color: accent3,
    nodeCount: Math.min(inputDim, 10)
  };
}

function getOutputColumn(onnxInfo, colorUnsat) {
  const output = (onnxInfo.outputs || [])[0];
  if (!output) {
    return { type: 'OUTPUT', name: 'Y', shape: '[]', color: colorUnsat, nodeCount: 3 };
  }
  const outputDim = getDimension(output.shape);
  return {
    type: 'OUTPUT',
    name: output.name,
    shape: `[${output.shape.join(', ')}]`,
    color: colorUnsat,
    nodeCount: Math.min(outputDim, 10)
  };
}

function makeLayerNode(layer, accent1, colorSat, hiddenNodeCount) {
  const op = layer.op.toUpperCase();
  return {
    type: op,
    name: layer.name || `layer_${layer.op.toLowerCase()}`,
    shape: 'Operation',
    color: op === 'RELU' ? colorSat : accent1,
    nodeCount: hiddenNodeCount
  };
}

function getLayerColumns(onnxInfo, accent1, colorSat, textSecondary, hiddenNodeCount) {
  if (!onnxInfo.layers) return [];
  const visibleOps = ['gemm', 'conv', 'relu', 'flatten', 'add', 'linear', 'matmul', 'sigmoid', 'tanh'];
  const filtered = onnxInfo.layers.filter(l => visibleOps.includes(l.op.toLowerCase()));

  if (filtered.length <= 5) {
    return filtered.map(l => makeLayerNode(l, accent1, colorSat, hiddenNodeCount));
  }

  const firstThree = filtered.slice(0, 3).map(l => makeLayerNode(l, accent1, colorSat, hiddenNodeCount));
  const placeholder = {
    type: '...',
    name: 'hidden_ops',
    shape: 'Skipped layers',
    color: textSecondary,
    nodeCount: 1,
    isPlaceholder: true
  };
  const lastLayer = makeLayerNode(filtered[filtered.length - 1], accent1, colorSat, hiddenNodeCount);
  return [...firstThree, placeholder, lastLayer];
}

function getTopologyColumns(onnxInfo, styles) {
  const accent1 = getStyleToken(styles, '--accent-1', 'oklch(53% 0.07 80)');
  const accent3 = getStyleToken(styles, '--accent-3', 'oklch(48% 0.14 250)');
  const colorSat = getStyleToken(styles, '--color-sat', 'oklch(56% 0.16 70)');
  const colorUnsat = getStyleToken(styles, '--color-unsat', 'oklch(52% 0.14 140)');
  const textSecondary = getStyleToken(styles, '--text-secondary', 'oklch(38% 0.006 80)');

  const firstInput = (onnxInfo.inputs || [])[0];
  const inputDim = firstInput ? getDimension(firstInput.shape) : 3;
  const hiddenNodeCount = Math.min(Math.max(inputDim + 1, 3), 10);

  return [
    getInputColumn(onnxInfo, accent3),
    ...getLayerColumns(onnxInfo, accent1, colorSat, textSecondary, hiddenNodeCount),
    getOutputColumn(onnxInfo, colorUnsat)
  ];
}

function drawBezierConnection(ctx, x1, y1, x2, y2, pulseT, destColor, borderSubtle) {
  ctx.strokeStyle = borderSubtle;
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  
  const cp1x = x1 + (x2 - x1) * 0.45;
  const cp2x = x1 + (x2 - x1) * 0.55;
  ctx.bezierCurveTo(cp1x, y1, cp2x, y2, x2, y2);
  ctx.stroke();

  if (currentJobId !== null) {
    const offset = (x1 * 0.05 + y1 * 0.05) % 1;
    const t = (pulseT + offset) % 1;
    
    const mt = 1 - t;
    const bx = mt * mt * mt * x1 + 3 * mt * mt * t * cp1x + 3 * mt * t * t * cp2x + t * t * t * x2;
    const by = mt * mt * mt * y1 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t * t * t * y2;

    ctx.fillStyle = destColor;
    ctx.beginPath();
    ctx.arc(bx, by, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawPlaceholderNodes(ctx, colX, textSecondary) {
  ctx.fillStyle = textSecondary;
  ctx.beginPath();
  ctx.arc(colX, 120, 1.5, 0, Math.PI * 2);
  ctx.arc(colX, 125, 1.5, 0, Math.PI * 2);
  ctx.arc(colX, 130, 1.5, 0, Math.PI * 2);
  ctx.fill();
}

function drawNodeCircle(ctx, x, y, radius, fillColor, strokeColor) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fillColor;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
}

function getColumnDrawingStyles(hovered, col, textSecondary, accent1) {
  const isMulti = col.nodeCount > 1;
  const config = {
    true: {
      fill: accent1,
      font: 'bold 8.5px "IBM Plex Mono", monospace',
      radius: 7,
      fillColor: col.color
    },
    false: {
      fill: textSecondary,
      font: '500 8.5px "IBM Plex Mono", monospace',
      radius: 5.5,
      fillColor: 'oklch(100% 0 0)'
    }
  };
  const active = config[hovered];
  active.spacingY = isMulti ? 140 / (col.nodeCount - 1) : 0;
  active.startY = isMulti ? 55 : 125;
  return active;
}

function drawColumnNodes(ctx, col, colX, hovered, textSecondary, accent1) {
  const d = getColumnDrawingStyles(hovered, col, textSecondary, accent1);
  ctx.fillStyle = d.fill;
  ctx.font = d.font;
  ctx.textAlign = 'center';
  ctx.fillText(col.type, colX, 30);

  col.nodePoints = [];

  if (col.isPlaceholder) {
    col.nodePoints.push({ x: colX, y: d.startY });
    drawPlaceholderNodes(ctx, colX, textSecondary);
    return;
  }

  for (let j = 0; j < col.nodeCount; j++) {
    const nodeY = d.startY + j * d.spacingY;
    col.nodePoints.push({ x: colX, y: nodeY });
    drawNodeCircle(ctx, colX, nodeY, d.radius, d.fillColor, col.color);
  }
}

function getTruncatedName(name) {
  return name.length > 18 ? name.slice(0, 15) + '…' : name;
}

function drawTooltip(ctx, hoveredColIdx, columns, mouseX, mouseY, width, height, borderSubtle, textPrimary, textSecondary) {
  if (hoveredColIdx === -1) return;
  const col = columns[hoveredColIdx];

  const boxW = 160;
  const boxH = 54;
  
  let tx = mouseX + 15;
  let ty = mouseY + 15;
  if (tx + boxW > width) tx = mouseX - boxW - 15;
  if (ty + boxH > height) ty = mouseY - boxH - 15;

  ctx.fillStyle = 'oklch(100% 0 0)';
  ctx.strokeStyle = borderSubtle;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(tx, ty, boxW, boxH, 4);
  ctx.fill();
  ctx.stroke();

  ctx.font = 'bold 9px "IBM Plex Mono", monospace';
  ctx.fillStyle = textPrimary;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';

  const cleanName = getTruncatedName(col.name);
  ctx.fillText(`Layer: ${cleanName}`, tx + 10, ty + 10);
  
  ctx.font = '500 8.5px "IBM Plex Mono", monospace';
  ctx.fillStyle = textSecondary;
  ctx.fillText(`Type:  ${col.type}`, tx + 10, ty + 24);
  ctx.fillText(`Shape: ${col.shape}`, tx + 10, ty + 36);
}

function drawPlaceholderConnections(ctx, c1, c2, pulseT, borderSubtle) {
  const yCenter = 125;
  c1.nodePoints.forEach(p1 => {
    drawBezierConnection(ctx, p1.x, p1.y, c2.nodePoints[0].x, yCenter, pulseT, c2.color, borderSubtle);
  });
  c2.nodePoints.forEach(p2 => {
    drawBezierConnection(ctx, c1.nodePoints[0].x, yCenter, p2.x, p2.y, pulseT, c2.color, borderSubtle);
  });
}

function isDenseConnection(c2Type, idx, totalCols) {
  const denseTypes = ['MATMUL', 'GEMM', 'LINEAR'];
  return denseTypes.includes(c2Type) || idx === 0 || idx === totalCols - 2;
}

function drawBetweenColumns(ctx, c1, c2, idx, totalCols, pulseT, borderSubtle) {
  if (c1.isPlaceholder || c2.isPlaceholder) {
    drawPlaceholderConnections(ctx, c1, c2, pulseT, borderSubtle);
    return;
  }

  const isDense = isDenseConnection(c2.type, idx, totalCols);
  c1.nodePoints.forEach((p1, idx1) => {
    c2.nodePoints.forEach((p2, idx2) => {
      if (isDense || idx1 === idx2) {
        drawBezierConnection(ctx, p1.x, p1.y, p2.x, p2.y, pulseT, c2.color, borderSubtle);
      }
    });
  });
}

function drawConnections(ctx, columns, pulseT, borderSubtle) {
  for (let i = 0; i < columns.length - 1; i++) {
    const c1 = columns[i];
    const c2 = columns[i + 1];
    if (!c1.nodePoints || !c2.nodePoints) continue;
    drawBetweenColumns(ctx, c1, c2, i, columns.length, pulseT, borderSubtle);
  }
}

function findHoveredColumn(mouseX, spacing, columnCount) {
  for (let i = 0; i < columnCount; i++) {
    if (Math.abs(mouseX - (i + 1) * spacing) < 25) {
      return i;
    }
  }
  return -1;
}

function renderOnnxTopology(onnxInfo) {
  if (topologyAnimId) {
    cancelAnimationFrame(topologyAnimId);
    topologyAnimId = null;
  }

  if (!onnxInfo) {
    topologyContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No model loaded.</p>';
    return;
  }

  if (onnxInfo.error) {
    topologyContainer.innerHTML = `<p style="color: var(--color-error); font-size: 0.85rem; padding: var(--space-md);">${escapeHtml(onnxInfo.error)}</p>`;
    return;
  }

  const styles = getComputedStyle(document.documentElement);
  const accent1 = getStyleToken(styles, '--accent-1', 'oklch(53% 0.07 80)');
  const textPrimary = getStyleToken(styles, '--text-primary', 'oklch(18% 0.006 80)');
  const textSecondary = getStyleToken(styles, '--text-secondary', 'oklch(38% 0.006 80)');
  const borderSubtle = getStyleToken(styles, '--border-subtle', 'oklch(92% 0.004 80)');

  const columns = getTopologyColumns(onnxInfo, styles);

  topologyContainer.innerHTML = '<canvas id="topology-canvas" style="width: 100%; display: block;"></canvas>';
  const canvas = document.getElementById('topology-canvas');
  const ctx = canvas.getContext('2d');

  let hoveredColIdx = -1;
  let mouseX = 0;
  let mouseY = 0;

  function resizeCanvas() {
    const rect = topologyContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = 250 * dpr;
    canvas.style.height = '250px';
    ctx.scale(dpr, dpr);
  }

  resizeCanvas();

  const resizeObserver = new ResizeObserver(() => {
    if (canvas.isConnected) {
      resizeCanvas();
    }
  });
  resizeObserver.observe(topologyContainer);

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;

    const width = canvas.width / (window.devicePixelRatio || 1);
    const spacing = width / (columns.length + 1);

    const found = findHoveredColumn(mouseX, spacing, columns.length);
    if (found !== hoveredColIdx) {
      hoveredColIdx = found;
      canvas.style.cursor = found === -1 ? 'default' : 'pointer';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    hoveredColIdx = -1;
  });

  let pulseT = 0;

  function animateTopology() {
    if (!canvas.isConnected) {
      resizeObserver.disconnect();
      return;
    }

    pulseT = (pulseT + 0.008) % 1;
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);

    ctx.clearRect(0, 0, width, height);

    const spacingX = width / (columns.length + 1);

    columns.forEach((col, colIdx) => {
      const colX = (colIdx + 1) * spacingX;
      const nodeCount = col.nodeCount;
      const spacingY = nodeCount > 1 ? 140 / (nodeCount - 1) : 0;
      const startY = nodeCount > 1 ? 55 : 125;
      
      col.nodePoints = [];
      for (let j = 0; j < nodeCount; j++) {
        col.nodePoints.push({ x: colX, y: startY + j * spacingY });
      }
    });

    drawConnections(ctx, columns, pulseT, borderSubtle);

    columns.forEach((col, colIdx) => {
      const colX = (colIdx + 1) * spacingX;
      drawColumnNodes(ctx, col, colX, colIdx === hoveredColIdx, textSecondary, accent1);
    });

    drawTooltip(ctx, hoveredColIdx, columns, mouseX, mouseY, width, height, borderSubtle, textPrimary, textSecondary);

    topologyAnimId = requestAnimationFrame(animateTopology);
  }

  animateTopology();
}

function buildBoundsRowsHTML(vnnlibInfo) {
  let boundsHTML = '';
  vnnlibInfo.variables.forEach((varName, idx) => {
    const b = vnnlibInfo.bounds[varName] || { lower: null, upper: null };
    const lowerStr = b.lower !== null ? b.lower.toFixed(4) : '-∞';
    const upperStr = b.upper !== null ? b.upper.toFixed(4) : '∞';

    boundsHTML += `
      <div class="bound-row" style="--i: ${idx}">
        <div class="bound-row__header">
          <span class="bound-row__name">${escapeHtml(varName)}</span>
          <span class="bound-row__values">[${lowerStr}, ${upperStr}]</span>
        </div>
        <div class="bound-row__bar">
          <div class="bound-row__fill"></div>
        </div>
      </div>
    `;
  });
  return boundsHTML;
}

function hasNoVariables(vnnlibInfo) {
  return !vnnlibInfo.variables || vnnlibInfo.variables.length === 0;
}

function renderVnnlibBounds(vnnlibInfo) {
  if (!vnnlibInfo) {
    boundsContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No specifications loaded.</p>';
    return;
  }

  if (vnnlibInfo.error) {
    boundsContainer.innerHTML = `<p style="color: var(--color-error); font-size: 0.85rem; padding: var(--space-md);">${escapeHtml(vnnlibInfo.error)}</p>`;
    return;
  }

  if (hasNoVariables(vnnlibInfo)) {
    boundsContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem; padding: var(--space-md);">No boundary variables found.</p>';
    return;
  }

  boundsContainer.innerHTML = buildBoundsRowsHTML(vnnlibInfo);
}



function stopTimers() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  if (logPollTimer) { clearTimeout(logPollTimer); logPollTimer = null; }
}

const RESULT_CONFIGS = {
  unsat: {
    label: '✅ UNSAT — Property Proved',
    className: 'result-badge--unsat',
    description: 'The neural network satisfies the specification. No counterexample exists within the defined input bounds.'
  },
  sat: {
    label: '⚠️ SAT — Property Disproved',
    className: 'result-badge--sat',
    description: 'The specification is violated. A counterexample input was found.'
  },
  timeout: {
    label: '⏱️ Timeout',
    className: 'result-badge--timeout',
    description: 'Verification did not complete within the time limit. Try increasing the timeout or using a GPU.'
  },
  unknown: {
    label: '❓ Unknown',
    className: 'result-badge--unknown',
    description: 'Verification finished but could not determine the result.'
  }
};

function buildMetadataRowHTML(label, value) {
  return `
    <div class="result-detail-row">
      <span class="result-detail-label">${label}</span>
      <span class="result-detail-value">${escapeHtml(value || '—')}</span>
    </div>
  `;
}

function buildDetailsHTML(data, description, result) {
  let html = `
    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: var(--space-md); line-height: 1.6;">${description}</p>
    <div class="result-detail-row">
      <span class="result-detail-label">Result</span>
      <span class="result-detail-value">${result}</span>
    </div>
  `;

  if (data.runtime != null) {
    html += `
      <div class="result-detail-row">
        <span class="result-detail-label">Runtime</span>
        <span class="result-detail-value">${formatTime(data.runtime)}</span>
      </div>
    `;
  }

  html += buildMetadataRowHTML('Network', data.net_name);
  html += buildMetadataRowHTML('Specification', data.spec_name);
  html += buildMetadataRowHTML('Device', data.device);

  if (data.counterexample) {
    html += `
      <details class="result-cex">
        <summary>Counterexample</summary>
        <pre>${escapeHtml(data.counterexample)}</pre>
      </details>
    `;
  }

  return html;
}

function showResult(data) {
  showState('completed');

  const result = (data.result || 'unknown').toLowerCase();
  const config = RESULT_CONFIGS[result] || RESULT_CONFIGS.unknown;

  resultBadge.textContent = config.label;
  resultBadge.className = `result-badge ${config.className}`;
  resultDetails.innerHTML = buildDetailsHTML(data, config.description, result);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function showState(state) {
  stateIdle.hidden = state !== 'idle';
  stateRunning.hidden = state !== 'running';
  stateCompleted.hidden = state !== 'completed';
  stateError.hidden = state !== 'error';

  const visible = {
    idle: stateIdle,
    running: stateRunning,
    completed: stateCompleted,
    error: stateError,
  }[state];

  if (visible) visible.classList.add('fade-in');
}

function renderLoadingPlaceholders() {
  examplesGrid.innerHTML = Array(3).fill(0).map(() => `
    <div class="example-card example-card--loading">
      <div class="example-card__name">Loading…</div>
      <div class="example-card__desc">Fetching examples from server</div>
    </div>
  `).join('');
}

function attachExampleCardHandlers() {
  examplesGrid.querySelectorAll('.example-card').forEach((card) => {
    card.addEventListener('click', () => loadExample(card.dataset.net, card.dataset.spec));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        loadExample(card.dataset.net, card.dataset.spec);
      }
    });
  });
}

function renderExamplesList(quickExamples) {
  examplesGrid.innerHTML = quickExamples.map((ex, idx) => `
    <div class="example-card" tabindex="0" role="button"
         data-net="${escapeHtml(ex.net)}" data-spec="${escapeHtml(ex.spec)}"
         style="--i: ${idx}">
      <div class="example-card__name">${escapeHtml(ex.name)}</div>
      <div class="example-card__desc">${escapeHtml(ex.description)}</div>
      <div class="example-card__files">
        <span class="example-card__file">${escapeHtml(ex.net)}</span>
        <span class="example-card__file">${escapeHtml(ex.spec)}</span>
      </div>
    </div>
  `).join('');

  attachExampleCardHandlers();
}

function hasNoExamples(data) {
  return !data.quick_examples || data.quick_examples.length === 0;
}

async function loadExamples() {
  renderLoadingPlaceholders();

  try {
    const res = await fetch(`${API_BASE}/examples`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (hasNoExamples(data)) {
      examplesGrid.innerHTML = '<p style="color: var(--text-muted);">No examples available. Start the backend server to load examples.</p>';
      return;
    }

    renderExamplesList(data.quick_examples);

  } catch (err) {
    console.warn('Could not load examples:', err);
    examplesGrid.innerHTML = `
      <p style="color: var(--text-muted);">
        Could not connect to the backend server. Examples will be available when the server is running.
      </p>
    `;
  }
}

async function fetchExampleFile(fileName) {
  const res = await fetch(`${API_BASE}/example/${fileName}`);
  if (!res.ok) throw new Error(`Failed to fetch ${fileName}`);
  const blob = await res.blob();
  return new File([blob], fileName, { type: 'application/octet-stream' });
}

function updateUploadZoneWithFile(zone, fileInfoEl, filenameEl, filesizeEl, file) {
  zone.classList.add('has-file');
  zone.querySelector('.upload-zone__content').hidden = true;
  fileInfoEl.hidden = false;
  filenameEl.textContent = file.name;
  filesizeEl.textContent = formatBytes(file.size);
}

async function loadExample(netName, specName) {
  if (currentJobId) return;

  showToast('Loading example files…', 'info');

  try {
    const [netFile, specFile] = await Promise.all([
      fetchExampleFile(netName),
      fetchExampleFile(specName)
    ]);

    onnxFile = netFile;
    updateUploadZoneWithFile(onnxZone, onnxFileInfo, onnxFilename, onnxFilesize, netFile);

    vnnlibFile = specFile;
    updateUploadZoneWithFile(vnnlibZone, vnnlibFileInfo, vnnlibFilename, vnnlibFilesize, specFile);

    updateVerifyButton();
    showToast('Example loaded! Click Verify to start.', 'success');

    document.getElementById('upload-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    showToast(`Error loading example: ${err.message}`, 'error');
  }
}

function showToast(message, type = 'info') {
  // Remove existing toasts
  document.querySelectorAll('.toast').forEach((t) => t.remove());

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    padding: 12px 24px;
    border-radius: 12px;
    font-size: 0.875rem;
    font-weight: 500;
    font-family: var(--font-body);
    z-index: 1000;
    opacity: 0;
    transition: all 0.3s var(--ease-out);
    backdrop-filter: blur(12px);
    max-width: 90vw;
    text-align: center;
    border: 1px solid;
  `;

  const colors = {
    info: { bg: 'rgba(99, 102, 241, 0.15)', border: 'rgba(99, 102, 241, 0.3)', color: '#a5b4fc' },
    success: { bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.3)', color: '#6ee7b7' },
    error: { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.3)', color: '#fca5a5' },
    warning: { bg: 'rgba(245, 158, 11, 0.15)', border: 'rgba(245, 158, 11, 0.3)', color: '#fcd34d' },
  };

  const c = colors[type] || colors.info;
  toast.style.background = c.bg;
  toast.style.borderColor = c.border;
  toast.style.color = c.color;

  document.body.appendChild(toast);

  requestAnimationFrame(() => {
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(-50%) translateY(0)';
  });

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(-50%) translateY(20px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function isModifierActive(e) {
  return e.ctrlKey || e.metaKey;
}

// Keyboard Shortcuts
function handleVerifyShortcut(e) {
  if (e.key !== 'Enter') return;
  if (!isModifierActive(e)) return;
  if (verifyBtn.disabled) return;

  e.preventDefault();
  startVerification();
}

function handleCancelShortcut(e) {
  if (e.key !== 'Escape') return;
  if (!currentJobId) return;
  if (cancelBtn.disabled) return;

  e.preventDefault();
  cancelBtn.click();
}

// Init
loadExamples();

document.addEventListener('keydown', (e) => {
  handleVerifyShortcut(e);
  handleCancelShortcut(e);
});

console.log(
  '%c%s%c\nDeveloped by GMU, NII, HUST, and UVA.\nReport bugs or check source at: https://github.com/dynaroars/neuralsat',
  'color: #677a56; font-family: "IBM Plex Mono", monospace; font-size: 14px; font-weight: bold;',
  'NeuralSAT Verification Bench',
  'color: inherit; font-family: sans-serif;'
);

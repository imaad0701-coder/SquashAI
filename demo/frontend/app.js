// Demo frontend. Plain JS, no build step, no framework. Talks to
// POST /api/analyze and GET /api/history[/{id}] and renders exactly what
// they return: ShotPipeline's per-frame debug_report, phases (an
// AnalysisResult), findings (a FindingsReport), the upload pre-flight checks,
// and the backend's list of outcome states. Nothing is inferred client-side:
// which states apply comes from the server.
//
// Phase boundaries are unvalidated (see docs/STATUS.md) and rendered
// accordingly: derivation_method and search_range_widened are always shown
// distinctly, never hidden -- same principle as the low-confidence-landmark
// treatment in drawSkeleton(). Findings follow it too: reported (solid),
// suppressed (dimmed + hatched, with the failed gate's observed vs required
// values), not applicable (outline). They are rendered as structured facts --
// numbers and counts -- never as coaching prose or verdict adjectives.

const PHASE_ORDER = ["prep", "backswing", "forward_swing", "follow_through", "recovery"];
const PHASE_COLORS = { detected: "#4da3ff", architectural: "#9aa1ac", unreliable: "#d98a3d" };
const CONTACT_COLOR = "#b57bef";
const WIDENED_RING_COLOR = "#e0574a";

const BONES = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"], ["left_ankle", "left_foot_index"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"], ["right_ankle", "right_foot_index"],
];

const videoInput = document.getElementById("video-input");
const shotTypeSelect = document.getElementById("shot-type");
const handednessSelect = document.getElementById("handedness");
const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const appEl = document.getElementById("app");
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const thresholdLabel = document.getElementById("threshold-label");
const frameIndexEl = document.getElementById("frame-index");
const frameTotalEl = document.getElementById("frame-total");
const frameTimeEl = document.getElementById("frame-time");
const anglesBody = document.querySelector("#angles-table tbody");
const kinematicsBody = document.querySelector("#kinematics-table tbody");
const postureBody = document.querySelector("#posture-table tbody");
const phaseTimeline = document.getElementById("phase-timeline");
const phaseTimelineCtx = phaseTimeline.getContext("2d");
const phaseTooltip = document.getElementById("phase-tooltip");
const phaseContactRuleEl = document.getElementById("phase-contact-rule");
const statesEl = document.getElementById("states");
const preflightPanel = document.getElementById("preflight-panel");
const preflightBody = document.querySelector("#preflight-table tbody");
const preflightSummary = document.getElementById("preflight-summary");
const findingsList = document.getElementById("findings-list");
const findingsSummary = document.getElementById("findings-summary");
const historyList = document.getElementById("history-list");
const videoUnavailable = document.getElementById("video-unavailable");
const videoPanelBits = [video, canvas, document.getElementById("skeleton-legend"), document.getElementById("phase-timeline-wrap")];
const sidebar = document.getElementById("sidebar");

let selectedFile = null;
let debugReport = null;
let visibilityThreshold = 0.5;
let rafHandle = null;
let phaseTimelineTicks = []; // [{x, frameIndex, kind, label, derivationMethod, widened}]
let phaseScrubbing = false;

videoInput.addEventListener("change", () => {
  selectedFile = videoInput.files[0] || null;
  analyzeBtn.disabled = !selectedFile;
});

analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  statusEl.textContent = "Uploading and running the pipeline -- this can take a while for longer clips...";

  const form = new FormData();
  form.append("video", selectedFile);
  form.append("shot_type", shotTypeSelect.value);
  form.append("handedness", handednessSelect.value);

  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: form });
    const body = await resp.json().catch(() => null);
    if (!body || (!resp.ok && !body.states)) {
      // A plain HTTP error (e.g. a bad form field) rather than one of the outcome states.
      throw new Error((body && body.detail) || `HTTP ${resp.status}`);
    }
    if (!resp.ok) {
      // upload_rejected / pipeline_failed: no per-frame data to show.
      debugReport = null;
      appEl.hidden = true;
      renderOutcome(body);
      statusEl.textContent = body.states.includes("upload_rejected") ? "Upload rejected." : "Pipeline failed.";
      return;
    }

    debugReport = body;
    visibilityThreshold = debugReport.visibility_threshold ?? 0.5;
    thresholdLabel.textContent = visibilityThreshold.toFixed(2);
    frameTotalEl.textContent = debugReport.landmark_frames.length;
    phaseContactRuleEl.textContent = debugReport.phases ? `(contact_rule=${debugReport.phases.contact_rule})` : "";

    showVideoPanel(true);
    video.src = URL.createObjectURL(selectedFile);
    appEl.hidden = false;
    renderOutcome(debugReport);
    statusEl.textContent = `Done: ${debugReport.landmark_frames.length} frames processed.`;
  } catch (err) {
    statusEl.textContent = `Failed: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
    refreshHistory();
  }
}

// --- outcome states, pre-flight, findings, history ----------------------------

const STATE_TEXT = {
  upload_rejected: ["Upload rejected", "The file failed a pre-flight check and was not analysed."],
  pipeline_failed: ["Pipeline failed", "The tracking pipeline raised an error on this clip."],
  quality_warned: ["Quality warning", "The clip was analysed, but a pre-flight check flagged something that can reduce reliability."],
  phases_unavailable: ["Phases unavailable", "Phase detection failed on this clip, so findings that need swings could not be evaluated."],
  all_findings_suppressed: ["No findings reported", "Every finding was suppressed or not applicable. The reasons are listed per finding below."],
  low_overall_confidence: ["Low overall tracking confidence", ""],
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderOutcome(data) {
  renderStates(data);
  renderPreflight(data.preflight);
  renderFindings(data.findings, data.findings_error);
}

function renderStates(data) {
  statesEl.innerHTML = "";
  for (const state of data.states || []) {
    const [title, text] = STATE_TEXT[state] || [state, ""];
    const box = el("div", `state state-${state}`);
    box.appendChild(el("strong", "", title));
    let detail = text;
    if (state === "upload_rejected" || state === "pipeline_failed") detail = data.detail || text;
    if (state === "phases_unavailable" && data.phases_error) detail = `${text} (${data.phases_error})`;
    if (state === "low_overall_confidence") {
      detail = `${fmt(100 * data.tracking_coverage, 0)}% of joint-angle samples are valid with confidence ` +
        `>= the visibility gate (display threshold ${fmt(100 * data.low_confidence_threshold, 0)}%, a demo UI ` +
        `choice, not a validated bar). Treat every number on this page with extra caution.`;
    }
    box.appendChild(el("span", "", detail));
    if (state === "quality_warned" && data.preflight) {
      const ul = el("ul");
      for (const c of data.preflight.checks.filter((c) => c.status === "warn")) ul.appendChild(el("li", "", c.message));
      box.appendChild(ul);
    }
    statesEl.appendChild(box);
  }
  statesEl.hidden = !(data.states || []).length;
}

function renderPreflight(pf) {
  preflightBody.innerHTML = "";
  preflightPanel.hidden = !pf;
  if (!pf) return;
  const counts = {};
  for (const c of pf.checks) {
    counts[c.status] = (counts[c.status] || 0) + 1;
    const tr = el("tr");
    tr.appendChild(el("td", "", c.name));
    tr.appendChild(el("td", `pf-${c.status}`, c.status));
    tr.appendChild(el("td", "", c.observed ?? "--"));
    tr.appendChild(el("td", "", c.limit ?? "--"));
    tr.appendChild(el("td", "", c.message));
    preflightBody.appendChild(tr);
  }
  preflightSummary.textContent = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(" · ");
}

function n(x, digits = 1) { return typeof x === "number" ? x.toFixed(digits) : "--"; }

function gateText(g) {
  return `${g.gate}: observed ${g.observed}, required ${g.comparator} ${g.required}`;
}

function factLines(f) {
  const v = f.value, u = f.unit;
  const digits = u === "ratio" || u === "shoulder widths" ? 3 : 1;
  if (f.kind === "consistency") {
    const lines = [
      `repetitions: ${v.repetitions}`,
      `mean: ${n(v.mean, digits)} ${u}`,
      `standard deviation across swings: ${n(v.std, digits)} ${u}`,
      `range: ${n(v.minimum, digits)} to ${n(v.maximum, digits)} ${u}`,
    ];
    if (v.noise_floor !== null) {
      const rel = v.noise_floor_comparison === "above_noise_floor" ? "larger than" : "not larger than";
      lines.push(`estimated measurement noise in this clip: ${n(v.noise_floor, digits)} ${u} (the spread is ${rel} this)`);
    }
    return lines;
  }
  if (f.kind === "asymmetry") {
    const roles = v.racket_side ? ` (racket side: ${v.racket_side})` : "";
    return [
      `repetitions: ${v.repetitions}${roles}`,
      `left mean: ${n(v.left_mean)} ${u}; right mean: ${n(v.right_mean)} ${u}`,
      `mean paired difference (left minus right): ${n(v.mean_difference)} ${u}, SD ${n(v.difference_std)} ${u}`,
    ];
  }
  return [
    `repetitions: ${v.repetitions}`,
    `elbow peak first: ${v.proximal_first_count}; wrist peak first: ${v.distal_first_count}; same frame: ${v.same_frame_count}`,
    `median lag (wrist peak minus elbow peak): ${n(v.median_lag_ms)} ms; one frame = ${n(v.ms_per_frame)} ms`,
  ];
}

function renderFinding(f) {
  const cls = { reported: "finding-reported", suppressed: "finding-suppressed", not_applicable: "finding-na" }[f.outcome];
  const card = el("div", `finding ${cls}`);
  const head = (parent) => {
    parent.appendChild(el("div", "f-title", f.description));
    parent.appendChild(el("div", "f-id", `${f.rule_id} · ${f.outcome.replace("_", " ")}`));
  };
  if (f.outcome === "reported") {
    head(card);
    const ul = el("ul", "f-facts");
    for (const line of factLines(f)) ul.appendChild(el("li", "", line));
    if (f.swing_exclusions.length) {
      ul.appendChild(el("li", "", `excluded swings: ${f.swing_exclusions.map((e) => `#${e.swing_index + 1} (${gateText(e.check)})`).join("; ")}`));
    }
    card.appendChild(ul);
  } else if (f.outcome === "suppressed") {
    const details = el("details");
    const summary = el("summary");
    head(summary);
    summary.appendChild(el("div", "f-reason", f.gate_failures.map(gateText).join("; ")));
    details.appendChild(summary);
    const ul = el("ul", "f-facts");
    for (const e of f.swing_exclusions) ul.appendChild(el("li", "", `swing #${e.swing_index + 1}: ${gateText(e.check)}`));
    details.appendChild(ul);
    card.appendChild(details);
  } else {
    head(card);
    card.appendChild(el("div", "f-reason", f.not_applicable_reason));
  }
  return card;
}

function renderFindings(report, error) {
  findingsList.innerHTML = "";
  if (!report) {
    findingsSummary.textContent = error ? `(not evaluated: ${error})` : "";
    return;
  }
  const count = (o) => report.findings.filter((f) => f.outcome === o).length;
  findingsSummary.textContent = `(rule set ${report.rule_set_version} · ${report.swing_count} swing(s) · ` +
    `${count("reported")} reported, ${count("suppressed")} suppressed, ${count("not_applicable")} not applicable)`;
  for (const kind of ["consistency", "asymmetry", "sequencing"]) {
    const group = el("div", "finding-group");
    group.appendChild(el("h4", "", kind));
    for (const f of report.findings.filter((f) => f.kind === kind)) group.appendChild(renderFinding(f));
    findingsList.appendChild(group);
  }
}

function showVideoPanel(available) {
  for (const node of videoPanelBits) node.hidden = !available;
  videoUnavailable.hidden = available;
  sidebar.hidden = !available;
}

async function refreshHistory(activeId) {
  const resp = await fetch("/api/history").catch(() => null);
  if (!resp || !resp.ok) return;
  const items = await resp.json();
  historyList.innerHTML = "";
  for (const item of items) {
    const li = el("li");
    if (item.analysis_id === activeId) li.classList.add("active");
    li.appendChild(el("span", "h-name", item.filename || "(unnamed)"));
    const when = new Date(item.created_at * 1000).toLocaleTimeString();
    const status = item.states.includes("upload_rejected") ? "rejected"
      : item.states.includes("pipeline_failed") ? "failed" : `${item.reported} finding(s) reported`;
    li.appendChild(el("span", "h-meta", `${when} · ${item.shot_type} · ${status}`));
    li.addEventListener("click", () => openHistory(item.analysis_id));
    historyList.appendChild(li);
  }
}

async function openHistory(id) {
  const resp = await fetch(`/api/history/${id}`);
  if (!resp.ok) { statusEl.textContent = "That entry is no longer in this session's history."; refreshHistory(); return; }
  const entry = await resp.json();
  debugReport = null;
  video.removeAttribute("src");
  phaseTimelineTicks = [];
  phaseTimelineCtx.clearRect(0, 0, phaseTimeline.width, phaseTimeline.height);
  showVideoPanel(false);
  phaseContactRuleEl.textContent = "";
  appEl.hidden = !(entry.findings || entry.phases_error || entry.findings_error);
  renderOutcome(entry);
  statusEl.textContent = `Showing stored results for ${entry.filename} (video and per-frame data not kept).`;
  refreshHistory(id);
}

refreshHistory();

video.addEventListener("loadedmetadata", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  phaseTimeline.width = phaseTimeline.clientWidth;
  phaseTimeline.height = 54;
  buildPhaseTimelineTicks();
  drawPhaseTimeline(0);
});

video.addEventListener("play", () => {
  cancelAnimationFrame(rafHandle);
  const step = () => {
    renderCurrentFrame();
    if (!video.paused && !video.ended) rafHandle = requestAnimationFrame(step);
  };
  step();
});
video.addEventListener("pause", () => cancelAnimationFrame(rafHandle));
video.addEventListener("seeked", renderCurrentFrame);

function nearestFrameIndex(currentTimeMs) {
  const frames = debugReport.landmark_frames;
  let lo = 0, hi = frames.length - 1;
  if (currentTimeMs <= frames[0].timing.timestamp_ms) return 0;
  if (currentTimeMs >= frames[hi].timing.timestamp_ms) return hi;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].timing.timestamp_ms < currentTimeMs) lo = mid + 1;
    else hi = mid;
  }
  const a = frames[Math.max(0, lo - 1)], b = frames[lo];
  return Math.abs(a.timing.timestamp_ms - currentTimeMs) <= Math.abs(b.timing.timestamp_ms - currentTimeMs)
    ? a.timing.frame_index : b.timing.frame_index;
}

function renderCurrentFrame() {
  if (!debugReport) return;
  const idx = nearestFrameIndex(video.currentTime * 1000);
  const frame = debugReport.landmark_frames[idx];
  if (!frame) return;

  frameIndexEl.textContent = frame.timing.frame_index;
  frameTimeEl.textContent = `(${(frame.timing.timestamp_ms / 1000).toFixed(2)}s)`;

  drawSkeleton(frame);
  fillAnglesTable(idx);
  fillKinematicsTable(idx);
  fillPostureTable(idx);
  drawPhaseTimeline(frame.timing.frame_index);
}

function frameTotal() {
  return debugReport.landmark_frames.length;
}

function frameToTimelineX(frameIndex) {
  const w = phaseTimeline.width;
  const total = Math.max(1, frameTotal() - 1);
  return (frameIndex / total) * w;
}

function timelineXToFrame(clientX) {
  const rect = phaseTimeline.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  return Math.round(fraction * (frameTotal() - 1));
}

// Builds the tick list once per analysis run (frame positions don't change
// while scrubbing) -- drawPhaseTimeline() just redraws from this every frame.
function buildPhaseTimelineTicks() {
  phaseTimelineTicks = [];
  if (!debugReport || !debugReport.phases) return;
  for (const swing of debugReport.phases.swings) {
    for (const phase of PHASE_ORDER) {
      const boundary = swing[phase];
      if (boundary.frame_index === null || boundary.frame_index === undefined) continue;
      phaseTimelineTicks.push({
        frameIndex: boundary.frame_index,
        kind: "boundary",
        label: phase,
        derivationMethod: boundary.derivation_method,
        widened: boundary.search_range_widened,
      });
    }
    if (swing.contact_frame !== null && swing.contact_frame !== undefined) {
      phaseTimelineTicks.push({ frameIndex: swing.contact_frame, kind: "contact", label: "contact", widened: false });
    }
  }
}

function drawPhaseTimeline(currentFrameIndex) {
  const w = phaseTimeline.width, h = phaseTimeline.height;
  phaseTimelineCtx.clearRect(0, 0, w, h);
  if (!debugReport) return;

  // track
  phaseTimelineCtx.strokeStyle = "#2c3038";
  phaseTimelineCtx.lineWidth = 2;
  phaseTimelineCtx.beginPath();
  phaseTimelineCtx.moveTo(4, h / 2);
  phaseTimelineCtx.lineTo(w - 4, h / 2);
  phaseTimelineCtx.stroke();

  for (const tick of phaseTimelineTicks) {
    const x = frameToTimelineX(tick.frameIndex);
    const y = h / 2;
    const r = tick.kind === "contact" ? 5 : 4.5;

    if (tick.kind === "contact") {
      phaseTimelineCtx.beginPath();
      phaseTimelineCtx.arc(x, y, r, 0, Math.PI * 2);
      phaseTimelineCtx.fillStyle = CONTACT_COLOR;
      phaseTimelineCtx.fill();
    } else if (tick.derivationMethod === "architectural") {
      // Hollow, dashed ring -- "not measured," distinct from a filled dot.
      phaseTimelineCtx.beginPath();
      phaseTimelineCtx.arc(x, y, r, 0, Math.PI * 2);
      phaseTimelineCtx.setLineDash([2, 2]);
      phaseTimelineCtx.strokeStyle = PHASE_COLORS.architectural;
      phaseTimelineCtx.lineWidth = 1.5;
      phaseTimelineCtx.stroke();
      phaseTimelineCtx.setLineDash([]);
    } else if (tick.derivationMethod === "unreliable") {
      // Filled but hatched, same visual family as a low-confidence landmark.
      phaseTimelineCtx.save();
      phaseTimelineCtx.beginPath();
      phaseTimelineCtx.arc(x, y, r, 0, Math.PI * 2);
      phaseTimelineCtx.fillStyle = "#3a2a15";
      phaseTimelineCtx.fill();
      phaseTimelineCtx.clip();
      phaseTimelineCtx.strokeStyle = PHASE_COLORS.unreliable;
      phaseTimelineCtx.lineWidth = 1;
      for (let d = -r; d <= r; d += 2.5) {
        phaseTimelineCtx.beginPath();
        phaseTimelineCtx.moveTo(x - r + d, y - r);
        phaseTimelineCtx.lineTo(x + r + d, y + r);
        phaseTimelineCtx.stroke();
      }
      phaseTimelineCtx.restore();
    } else {
      phaseTimelineCtx.beginPath();
      phaseTimelineCtx.arc(x, y, r, 0, Math.PI * 2);
      phaseTimelineCtx.fillStyle = PHASE_COLORS.detected;
      phaseTimelineCtx.fill();
    }

    if (tick.widened) {
      phaseTimelineCtx.beginPath();
      phaseTimelineCtx.arc(x, y, r + 3, 0, Math.PI * 2);
      phaseTimelineCtx.strokeStyle = WIDENED_RING_COLOR;
      phaseTimelineCtx.lineWidth = 1.5;
      phaseTimelineCtx.stroke();
    }
  }

  // playhead
  if (typeof currentFrameIndex === "number") {
    const x = frameToTimelineX(currentFrameIndex);
    phaseTimelineCtx.beginPath();
    phaseTimelineCtx.moveTo(x, 2);
    phaseTimelineCtx.lineTo(x, h - 2);
    phaseTimelineCtx.strokeStyle = "#e6e8eb";
    phaseTimelineCtx.lineWidth = 1;
    phaseTimelineCtx.stroke();
  }
}

function seekToTimelineEvent(evt) {
  if (!debugReport) return;
  const frameIndex = timelineXToFrame(evt.clientX);
  const frame = debugReport.landmark_frames.find((f) => f.timing.frame_index === frameIndex) || debugReport.landmark_frames[frameIndex];
  if (frame) video.currentTime = frame.timing.timestamp_ms / 1000;
}

phaseTimeline.addEventListener("mousedown", (evt) => { phaseScrubbing = true; seekToTimelineEvent(evt); });
window.addEventListener("mouseup", () => { phaseScrubbing = false; });
phaseTimeline.addEventListener("mousemove", (evt) => {
  if (phaseScrubbing) { seekToTimelineEvent(evt); return; }
  if (!phaseTimelineTicks.length) { phaseTooltip.style.opacity = 0; return; }
  const rect = phaseTimeline.getBoundingClientRect();
  const mouseX = evt.clientX - rect.left;
  let nearest = null, nearestDist = Infinity;
  for (const tick of phaseTimelineTicks) {
    const x = frameToTimelineX(tick.frameIndex);
    const dist = Math.abs(x - mouseX);
    if (dist < nearestDist) { nearestDist = dist; nearest = tick; }
  }
  if (nearest && nearestDist <= 8) {
    const bits = [`${nearest.label} @ frame ${nearest.frameIndex}`];
    if (nearest.kind === "boundary") bits.push(nearest.derivationMethod);
    if (nearest.widened) bits.push("range widened");
    phaseTooltip.textContent = bits.join(" · ");
    phaseTooltip.style.left = `${(frameToTimelineX(nearest.frameIndex) / phaseTimeline.width) * 100}%`;
    phaseTooltip.style.top = "0px";
    phaseTooltip.style.opacity = 1;
  } else {
    phaseTooltip.style.opacity = 0;
  }
});
phaseTimeline.addEventListener("mouseleave", () => { phaseTooltip.style.opacity = 0; });

function drawSkeleton(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const lm = frame.pose_landmarks;

  const isConfident = (l) => l && l.visibility >= visibilityThreshold;

  for (const [a, b] of BONES) {
    const la = lm[a], lb = lm[b];
    if (!la || !lb) continue;
    const confident = isConfident(la) && isConfident(lb);
    ctx.beginPath();
    ctx.moveTo(la.position.x, la.position.y);
    ctx.lineTo(lb.position.x, lb.position.y);
    ctx.strokeStyle = confident ? "#4da3ff" : "#7a7f88";
    ctx.lineWidth = 2;
    ctx.setLineDash(confident ? [] : [5, 4]);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  for (const [name, l] of Object.entries(lm)) {
    const confident = isConfident(l);
    const { x, y } = l.position;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = confident ? "#37d67a" : "#7a7f88";
    ctx.fill();
    if (!confident) {
      // Hatch pattern so a below-gate landmark reads as visually distinct,
      // never simply hidden -- the whole point of the reliability system.
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.clip();
      ctx.strokeStyle = "#3a3f47";
      ctx.lineWidth = 1;
      for (let d = -6; d <= 6; d += 3) {
        ctx.beginPath();
        ctx.moveTo(x - 6 + d, y - 6);
        ctx.lineTo(x + 6 + d, y + 6);
        ctx.stroke();
      }
      ctx.restore();
    }
  }
}

function fmt(n, digits = 1) {
  return typeof n === "number" ? n.toFixed(digits) : "--";
}

function fillAnglesTable(idx) {
  anglesBody.innerHTML = "";
  const entries = Object.entries(debugReport.angle_measurements).sort(([a], [b]) => a.localeCompare(b));
  for (const [key, series] of entries) {
    const m = series[idx];
    if (!m) continue;
    const tr = document.createElement("tr");
    if (!m.is_valid) tr.classList.add("invalid");
    tr.innerHTML = `<td>${key}</td><td>${m.is_valid ? fmt(m.angle_degrees) + "&deg;" : "invalid"}</td>` +
      `<td>${fmt(m.confidence, 2)}</td>`;
    anglesBody.appendChild(tr);
  }
}

function fillKinematicsTable(idx) {
  kinematicsBody.innerHTML = "";
  const entries = Object.entries(debugReport.kinematics).sort(([a], [b]) => a.localeCompare(b));
  for (const [key, metrics] of entries) {
    const tr = document.createElement("tr");
    let velCell = "--", accCell = "--";
    if (metrics.velocity) {
      const v = metrics.velocity[idx];
      velCell = v && v.is_valid && v.velocity
        ? fmt(Math.hypot(v.velocity.x, v.velocity.y, v.velocity.z), 0) + " px/s" : "invalid";
      const a = metrics.acceleration && metrics.acceleration[idx];
      accCell = a && a.is_valid && a.acceleration
        ? fmt(Math.hypot(a.acceleration.x, a.acceleration.y, a.acceleration.z), 0) + " px/s²" : "invalid";
    } else if (metrics.angular_velocity) {
      const v = metrics.angular_velocity[idx];
      velCell = v && v.is_valid ? fmt(v.angular_velocity_degrees_per_second, 0) + " °/s" : "invalid";
      const a = metrics.angular_acceleration && metrics.angular_acceleration[idx];
      accCell = a && a.is_valid ? fmt(a.angular_acceleration_degrees_per_second_squared, 0) + " °/s²" : "invalid";
    }
    tr.innerHTML = `<td>${key}</td><td>${velCell}</td><td>${accCell}</td>`;
    kinematicsBody.appendChild(tr);
  }
}

function fillPostureTable(idx) {
  postureBody.innerHTML = "";
  const entries = Object.entries(debugReport.posture || {}).sort(([a], [b]) => a.localeCompare(b));
  for (const [key, series] of entries) {
    const m = series[idx];
    if (!m) continue;
    const tr = document.createElement("tr");
    if (!m.is_valid) tr.classList.add("invalid");
    let valueText = "invalid";
    if (m.is_valid) {
      if ("right_foot_ratio" in m) valueText = fmt(m.right_foot_ratio, 2);
      else if ("height_ratio" in m) valueText = fmt(m.height_ratio, 2);
      else if ("normalized_displacement" in m) valueText = fmt(m.normalized_displacement, 3);
    }
    tr.innerHTML = `<td>${key}</td><td>${valueText}</td><td>${m.confidence !== undefined ? fmt(m.confidence, 2) : "--"}</td>`;
    postureBody.appendChild(tr);
  }
}

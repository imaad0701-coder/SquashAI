// Minimal demo frontend. Plain JS, no build step, no framework -- matches
// the "not a product" scope of this whole demo. Talks to the single
// POST /api/analyze endpoint and renders exactly what it returns: the real
// landmark_frames/angle_measurements/kinematics/posture from ShotPipeline's
// debug_report, plus phases (an AnalysisResult) from engine.phases. No
// scores, no findings, no coaching text -- none of that exists in the
// response to render in the first place. Phase boundaries are unvalidated
// (see docs/STATUS.md) and rendered accordingly: derivation_method and
// search_range_widened are always shown distinctly, never hidden -- same
// principle as the low-confidence-landmark treatment in drawSkeleton(),
// applied to phase boundaries instead of pose landmarks.

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
const phaseUnavailableEl = document.getElementById("phase-unavailable");
const phaseUnavailableReasonEl = document.getElementById("phase-unavailable-reason");

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
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${resp.status}`);
    }
    debugReport = await resp.json();
    visibilityThreshold = debugReport.visibility_threshold ?? 0.5;
    thresholdLabel.textContent = visibilityThreshold.toFixed(2);
    frameTotalEl.textContent = debugReport.landmark_frames.length;

    if (debugReport.phases) {
      phaseContactRuleEl.textContent = `(contact_rule=${debugReport.phases.contact_rule})`;
      phaseUnavailableEl.hidden = true;
    } else {
      phaseContactRuleEl.textContent = "";
      phaseUnavailableReasonEl.textContent = debugReport.phases_error ? `: ${debugReport.phases_error}` : "";
      phaseUnavailableEl.hidden = false;
    }

    video.src = URL.createObjectURL(selectedFile);
    appEl.hidden = false;
    statusEl.textContent = `Done: ${debugReport.landmark_frames.length} frames processed.`;
  } catch (err) {
    statusEl.textContent = `Failed: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
}

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

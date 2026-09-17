const $ = (s) => document.querySelector(s),
  pct = (n) => (n * 100).toFixed(1) + "%",
  num = (n) => (n == null ? "—" : n.toFixed(3));
let run = null,
  candidate = null;
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
async function api(path, body) {
  const r = await fetch(
    "/api/" + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const d = await r.json();
  if (!r.ok)
    throw Error(
      typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail),
    );
  return d;
}
async function action(fn) {
  document.querySelectorAll("button").forEach((b) => (b.disabled = true));
  try {
    await fn();
    await status();
  } catch (e) {
    $("#message").textContent = e.message;
  } finally {
    $("#run").disabled = false;
    $("#rollback").disabled = false;
    $("#train").disabled = !run?.config.labels;
    $("#promote").disabled = !candidate?.eligible || candidate?.promoted;
  }
}
$("#severity").oninput = (e) =>
  ($("#severityText").textContent = pct(+e.target.value));
const lessons = {
  normal: [
    "Drift is a clue, not a verdict.",
    "The baseline model was trained on the same data-generating process. Some monitoring variation is expected. A replacement may fail the gate because there is no meaningful problem to fix.",
  ],
  sensor_shift: [
    "The inputs moved. Did performance?",
    "Temperature and vibration shift upward, but the relationship to failure remains the same. Drift alarms signal a change to investigate; they do not prove model degradation.",
  ],
  concept_shift: [
    "Same sensors. Different world.",
    "The link between vibration and failure reverses while sensor distributions stay stable. Input-only monitoring can miss this. Labeled outcomes reveal the damage, and a challenger can learn the new relationship.",
  ],
  missing_sensor: [
    "Missing data is a model problem.",
    "The vibration sensor drops readings. The pipeline imputes its training median, so predictions still run, but lost information can reduce quality. Retraining cannot recover a signal that was never measured.",
  ],
};
$("#run").onclick = () =>
  action(async () => {
    $("#message").textContent =
      "Generating telemetry and scoring with the active model…";
    run = await api("simulate", {
      scenario: $("#scenario").value,
      severity: +$("#severity").value,
      seed: +$("#seed").value,
      threshold: +$("#threshold").value,
      labels: $("#labels").checked,
    });
    candidate = null;
    render(run);
    $("#candidate").textContent =
      "Ready to train on the 900-row training partition. Gate and audit data remain separate.";
    $("#message").textContent =
      `Experiment ${run.hash} · ${run.version} · synthetic data · ${run.config.labels ? "labels available" : "outcomes hidden"}`;
  });
function render(d) {
  $("#f1").textContent = num(d.metrics?.f1);
  $("#ap").textContent = num(d.metrics?.average_precision);
  $("#brier").textContent = num(d.metrics?.brier);
  $("#alerts").textContent = d.drift.filter((x) => x.alert).length + " / 4";
  $("#version").textContent = d.version;
  $("#charts").innerHTML = d.drift
    .map((f) => {
      const max = Math.max(...f.reference, ...f.current, 0.01);
      return `<div class="chart"><div class="charthead"><b>${f.name}</b><span class="tag ${f.alert ? "alert" : "ok"}">${f.alert ? "INVESTIGATE" : "STABLE"}</span></div><svg viewBox="0 0 250 105" role="img" aria-label="${f.name} reference and current distributions">${f.reference.map((v, i) => `<rect x="${i * 15 + 4}" y="${95 - (v / max) * 88}" width="6" height="${(v / max) * 88}" fill="#627e94"/><rect x="${i * 15 + 10}" y="${95 - (f.current[i] / max) * 88}" width="6" height="${(f.current[i] / max) * 88}" fill="#6ce4bd"/>`).join("")}<line x1="0" x2="250" y1="96" y2="96" stroke="#354b57"/></svg><small>${f.range[0].toFixed(1)} – ${f.range[1].toFixed(1)} ${f.unit}<br>KS ${num(f.ks)} · p ${f.p == null ? "n/a" : f.p.toExponential(1)}<br>Missing ${pct(f.missing)}</small></div>`;
    })
    .join("");
  [$("#lessonTitle").textContent, $("#lesson").textContent] =
    lessons[d.config.scenario];
  $("#predictions").innerHTML =
    "<table><thead><tr><th>°C</th><th>mm/s</th><th>bar</th><th>Load %</th><th>Risk</th><th>Failure</th></tr></thead><tbody>" +
    d.examples
      .map(
        (e) =>
          `<tr>${e.values.map((v) => `<td>${v ?? "missing"}</td>`).join("")}<td>${pct(e.risk)}</td><td>${e.label ?? "hidden"}</td></tr>`,
      )
      .join("") +
    "</tbody></table>";
}
function renderCandidate(c) {
  $("#candidate").innerHTML =
    `<span class="tag ${c.eligible ? "ok" : "alert"}">${c.promoted ? "PROMOTED LOCALLY" : c.eligible ? "GATE PASSED" : "GATE FAILED"}</span><table><thead><tr><th>Gate · 300 rows</th><th>Champion</th><th>Challenger</th></tr></thead><tbody><tr><td>F1 ↑</td><td>${num(c.champion.f1)}</td><td>${num(c.challenger.f1)}</td></tr><tr><td>Brier ↓</td><td>${num(c.champion.brier)}</td><td>${num(c.challenger.brier)}</td></tr></tbody></table>${c.audit ? `<p>Post-selection audit · 300 untouched rows<br>F1: ${num(c.audit.champion.f1)} → ${num(c.audit.challenger.f1)}<br>Brier: ${num(c.audit.champion.brier)} → ${num(c.audit.challenger.brier)}</p>` : "<p>Audit results stay hidden until a passing candidate is selected for local promotion.</p>"}`;
}
$("#train").onclick = () =>
  action(async () => {
    candidate = await api("train", { run_id: run.id });
    renderCandidate(candidate);
    $("#message").textContent = candidate.eligible
      ? "The challenger passed the promotion gate. Inspect the comparison, then promote."
      : "The challenger did not pass the gate. The active model is unchanged.";
  });
$("#promote").onclick = () =>
  action(async () => {
    candidate = await api("promote", {
      run_id: run.id,
      candidate_id: candidate.id,
    });
    renderCandidate(candidate);
    $("#message").textContent =
      "Promoted in this local server session. Run a new experiment to score the new active model. Above monitoring metrics remain the original experiment.";
  });
$("#rollback").onclick = () =>
  action(async () => {
    await api("rollback", {});
    run = candidate = null;
    $("#candidate").textContent = "Baseline restored. Run a new experiment.";
    ["f1", "ap", "brier", "alerts"].forEach(
      (k) => ($("#" + k).textContent = "—"),
    );
    $("#charts").innerHTML =
      '<p class="empty">Baseline restored. Run a fresh experiment.</p>';
    $("#predictions").textContent = "Waiting for a new experiment.";
    $("#message").textContent =
      "Reset to baseline-v1. Session event history is retained.";
  });
async function status() {
  const s = await api("status");
  $("#version").textContent = s.version;
  $("#trail").innerHTML =
    s.audit_log
      .slice(-6)
      .reverse()
      .map(
        (e) =>
          `<div class="event">${esc(e.action.replaceAll("_", " "))}<span>${esc(e.version || e.candidate || e.scenario || "")} · ${esc(new Date(e.at).toLocaleTimeString())}${e.dataset_hash ? " · " + e.dataset_hash : ""}</span></div>`,
      )
      .join("") || "Run an experiment to start the trail.";
}
status().catch((e) => ($("#message").textContent = e.message));

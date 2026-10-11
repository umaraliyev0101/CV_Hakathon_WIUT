// Wizard controller: upload -> zones -> staff -> results. All state lives on the server (job id in
// the URL hash), so a page reload resumes where the job is.
import { ZoneEditor } from "/static/zones.js";
import { renderDashboard, h } from "/static/dashboard.js";

const $ = (id) => document.getElementById(id);
const STEPS = ["upload", "zones", "staff", "results"];
const state = { jobId: null, fromResults: false, changed: {}, people: [], stopDash: null, polling: 0 };

async function req(url, opts) {
  let res;
  try { res = await fetch(url, opts); } catch { throw new Error("Cannot reach the server. Is it still running?"); }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.error || `Request failed (${res.status})`);
  return body;
}
const putJson = (url, data) => req(url, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function notice(msg, isError = true) {
  const n = $("notice"); n.textContent = msg || ""; n.hidden = !msg; n.classList.toggle("error", !!msg && isError);
}
function showStep(name) {
  for (const s of STEPS) $(`step-${s}`).hidden = s !== name;
  const idx = STEPS.indexOf(name);
  document.querySelectorAll(".steps li").forEach((li) => {
    const i = STEPS.indexOf(li.dataset.step);
    li.classList.toggle("done", i < idx); li.classList.toggle("current", i === idx);
    if (i === idx) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0 });
}
function setJob(id) {
  state.jobId = id;
  location.hash = id ? `job=${id}` : "";
}

// ------------------------------------------------------------------ 1. upload + progress
function setProgress(frac, msg) {
  $("progress").hidden = false;
  const pct = Math.round(frac * 100);
  $("bar-fill").style.width = `${pct}%`;
  $("progress-msg").textContent = msg;
  $("progress-pct").textContent = `${pct}%`;
  document.querySelector("#progress .bar").setAttribute("aria-valuenow", String(pct));
}

$("upload-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  if (!file) { notice("Choose a video file first."); return; }
  notice("");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("preset", document.querySelector("input[name=preset]:checked").value);
  $("upload-btn").disabled = true;
  setProgress(0, "Uploading…");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) setProgress(ev.loaded / ev.total * 0.2, "Uploading…"); };
  xhr.onerror = () => { notice("Upload failed: cannot reach the server."); $("upload-btn").disabled = false; };
  xhr.onload = () => {
    let body = null; try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    if (xhr.status !== 200 || !body?.job_id) {
      notice(body?.error || `Upload failed (${xhr.status})`); $("upload-btn").disabled = false; return;
    }
    setJob(body.job_id); state.fromResults = false; pollJob();
  };
  xhr.send(fd);
});

async function pollJob() {
  const mine = ++state.polling;
  while (mine === state.polling) {
    let st;
    try { st = await req(`/api/jobs/${state.jobId}`); }
    catch {
      try { await new Promise((r) => setTimeout(r, 1000)); st = await req(`/api/jobs/${state.jobId}`); }   // one retry
      catch (err) { notice(err.message); $("upload-btn").disabled = false; return; }
    }
    if (mine !== state.polling) return;
    if (st.state === "failed") {
      notice(st.message || "Analysis failed."); $("upload-btn").disabled = false; showStep("upload"); return;
    }
    if (st.state === "ready_for_zones") { $("upload-btn").disabled = false; await openZones(); return; }
    if (st.state === "scored") { await openResults(); return; }
    showStep("upload");
    setProgress(0.2 + 0.8 * (st.progress || 0), st.message || "Analysing…");
    await new Promise((r) => setTimeout(r, 1000));
  }
}

// ------------------------------------------------------------------ 2. zones
const zoneEditor = new ZoneEditor({
  panel: $("step-zones"), img: $("zone-img"), canvas: $("zone-canvas"), list: $("zone-list"), form: $("zone-form"),
  name: $("zone-name"), kind: $("zone-kind"), hint: $("tool-hint"), error: $("zone-error"), floorRows: $("floor-rows"),
  btn: { select: $("tool-select"), rect: $("tool-rect"), poly: $("tool-poly"), finish: $("tool-finish"), del: $("tool-delete"), calib: $("tool-calib"), clearFloor: $("tool-clear-floor") },
});

async function openZones() {
  try {
    notice("");
    showStep("zones");
    const doc = await req(`/api/jobs/${state.jobId}/zones`);
    await zoneEditor.load(doc, `/api/jobs/${state.jobId}/keyframe.jpg?t=${Date.now()}`);
  } catch (err) { notice(err.message); }
}

$("zones-ok").addEventListener("click", async () => {
  const btn = $("zones-ok"); btn.disabled = true; zoneEditor.showError("");
  try {
    await putJson(`/api/jobs/${state.jobId}/zones`, zoneEditor.toDoc());
    if (state.fromResults) await scoreNow(); else await openStaff();
  } catch (err) { zoneEditor.showError(err.message); }
  finally { btn.disabled = false; }
});

// ------------------------------------------------------------------ 3. staff gallery
function personOrder(a, b) {
  const rank = (p) => (p.suggested_role === "staff" ? 0 : p.suggested_role === "unsure" ? 1 : 2);
  return rank(a) - rank(b) || (b.confidence ?? 0) - (a.confidence ?? 0) || a.person_id - b.person_id;
}

async function openStaff() {
  try {
    notice(""); showStep("staff");
    state.people = (await req(`/api/jobs/${state.jobId}/people`)).sort(personOrder);
    state.changed = {};
    renderPeople();
  } catch (err) { notice(err.message); }
}

const baseline = (p) => (p.suggested_role === "staff" ? "staff" : "customer");   // what the backend uses with no override
function effectiveRole(p) { return state.changed[p.person_id] || (p.role === "staff" ? "staff" : "customer"); }

function renderPeople() {
  const box = $("people"); box.textContent = "";
  if (!state.people.length) box.append(h("p", { class: "empty" }, "No people were detected in this video."));
  for (const p of state.people) {
    const role = effectiveRole(p);
    const sug = p.suggested_role || "customer";
    const card = h("div", { class: `person${role === "staff" ? " is-staff" : ""}`, role: "listitem" },
      p.thumb_url ? h("img", { src: p.thumb_url, alt: `Person ${p.person_id} (face blurred)`, loading: "lazy" })
                  : h("div", { class: "nothumb" }, "No preview"),
      h("div", {}, h("strong", {}, `Person ${p.person_id}`), " ",
        h("span", { class: `badge ${sug}` }, sug === "unsure" ? "unsure" : `suggested ${sug}`),
        p.confidence != null && h("small", {}, ` ${Math.round(p.confidence * 100)}%`)),
      (p.reasons || []).length ? h("ul", {}, p.reasons.slice(0, 3).map((r) => h("li", {}, r))) : null,
      h("button", { type: "button", "aria-pressed": String(role === "staff"), onclick: () => {
        state.changed[p.person_id] = role === "staff" ? "customer" : "staff"; renderPeople(); } },
      role === "staff" ? "Staff (click to change)" : "Customer (click to mark staff)"));
    box.append(card);
  }
  const n = state.people.filter((p) => effectiveRole(p) === "staff").length;
  $("staff-count").textContent = `${state.people.length} people found, ${n} marked as staff. Most likely staff are listed first.`;
}

async function finishStaff(skip) {
  const ok = $("staff-ok"), sk = $("staff-skip"); ok.disabled = sk.disabled = true;
  try {
    // PUT replaces all overrides, so send the FULL set: everyone whose chosen role differs from the suggestion
    const overrides = {};
    if (!skip) for (const p of state.people) if (effectiveRole(p) !== baseline(p)) overrides[p.person_id] = effectiveRole(p);
    await putJson(`/api/jobs/${state.jobId}/roles`, { overrides, skip });
    await scoreNow();
  } catch (err) { notice(err.message); }
  finally { ok.disabled = sk.disabled = false; }
}
$("staff-ok").addEventListener("click", () => finishStaff(false));
$("staff-skip").addEventListener("click", () => finishStaff(true));

// ------------------------------------------------------------------ 4. results
async function scoreNow() {
  notice("Scoring…", false);
  try {
    const report = await req(`/api/jobs/${state.jobId}/score`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    notice("");
    showResults(report);
  } catch (err) { notice(err.message); }
}
async function openResults() {
  try { showResults(await req(`/api/jobs/${state.jobId}/results`)); } catch (err) { notice(err.message); }
}
function showResults(report) {
  state.stopDash?.();
  showStep("results");
  state.stopDash = renderDashboard($("dashboard"), report, {
    jobId: state.jobId,
    getStatus: () => req(`/api/jobs/${state.jobId}`),
    onSales: async (file) => {
      const fd = new FormData(); fd.append("file", file);
      notice("Uploading sales…", false);
      try { await req(`/api/jobs/${state.jobId}/sales`, { method: "POST", body: fd }); notice(""); await openResults(); }
      catch (err) { notice(err.message); }
    },
    onEditZones: () => { state.stopDash?.(); state.fromResults = true; openZones(); },
    onEditRoles: () => { state.stopDash?.(); state.fromResults = true; openStaff(); },
  });
}

// ------------------------------------------------------------------ boot
showStep("upload");
const m = /job=([A-Za-z0-9_-]+)/.exec(location.hash);
if (m) { setJob(m[1]); pollJob(); }

"use strict";

// Same-origin requests go through Nginx; model inference stays in Python.
const API_BASE_URL = "/api";
const form = document.querySelector("#prediction-form");
const report = document.querySelector("#report");
const analyze = document.querySelector("#analyze");
const sample = document.querySelector("#sample");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const emptyState = document.querySelector("#empty-state");
const resultsPanel = document.querySelector("#results-panel");

function updateCount() {
  document.querySelector("#character-count").textContent = `${report.value.length.toLocaleString()} / 5,000`;
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function renderEntities(selector, entities) {
  const items = (entities.length ? entities : ["None returned"]).map((entity) => {
    const item = document.createElement("li");
    // Treat model output as text, never as HTML.
    item.textContent = entity;
    return item;
  });
  document.querySelector(selector).replaceChildren(...items);
}

function renderPrediction(prediction) {
  renderEntities("#ade-entities", prediction.entities.ADE);
  renderEntities("#drug-entities", prediction.entities.DRUG);
  document.querySelector("#severity-label").textContent = prediction.severity.label;
  const rows = Object.entries(prediction.severity.probabilities).map(([label, probability]) => {
    const row = document.createElement("div");
    row.className = "probability";
    const name = document.createElement("span");
    name.textContent = label;
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = probability;
    bar.setAttribute("aria-label", `${label} probability`);
    const value = document.createElement("output");
    value.textContent = `${(probability * 100).toFixed(1)}%`;
    row.append(name, bar, value);
    return row;
  });
  document.querySelector("#probabilities").replaceChildren(...rows);
  emptyState.hidden = true;
  results.hidden = false;
}

report.addEventListener("input", () => {
  report.setCustomValidity("");
  updateCount();
  results.hidden = true;
  emptyState.hidden = false;
  setStatus("Ready for your text.");
});

sample.addEventListener("click", () => {
  report.value = "Patient reported fever after Pfizer booster.";
  report.dispatchEvent(new Event("input"));
  report.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = report.value.trim();
  if (!text) {
    report.setCustomValidity("Enter some text before analyzing.");
    report.reportValidity();
    return;
  }

  analyze.disabled = true;
  sample.disabled = true;
  report.readOnly = true;
  analyze.textContent = "Analyzing…";
  resultsPanel.setAttribute("aria-busy", "true");
  results.hidden = true;
  emptyState.hidden = false;
  setStatus("Analyzing your report…");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    // fetch sends JSON over HTTP; await waits without blocking the browser UI.
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    if (!response.ok) {
      if (response.status === 422) {
        const errorBody = await response.json();
        const detail = errorBody.detail;
        const message = typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((item) => item.msg).join(" ")
            : "Check the text and try again.";
        throw new Error(message);
      }
      throw new Error(`Analysis failed (HTTP ${response.status}). Try again after checking the API.`);
    }
    const prediction = await response.json();
    renderPrediction(prediction);
    setStatus("Analysis complete. Results are from synthetic demo models.");
  } catch (error) {
    const message = error.name === "AbortError"
      ? "The request timed out. Check that the API is ready, then try again."
      : error instanceof TypeError
        ? "Cannot reach the API. Check the application connection and try again."
        : error.message;
    setStatus(message, true);
  } finally {
    clearTimeout(timeout);
    analyze.disabled = false;
    sample.disabled = false;
    report.readOnly = false;
    analyze.textContent = "Analyze text →";
    resultsPanel.setAttribute("aria-busy", "false");
  }
});

updateCount();

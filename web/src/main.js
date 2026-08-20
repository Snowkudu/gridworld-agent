import "./styles.css";

import { loadGridWorldModel } from "./model-runtime.js";
import { chooseAction } from "./policy.js";

const OBSERVATION_GRID = [
  0, 0, -1, 0, 0, 0, -1, 0, 0, 0,
  0, 1, -1, 0, -1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, -1, 0, -1, -1, 0, 0,
  -1, -1, 0, 0, 0, 0, 0, -1, 0, 0,
  0, 0, 0, -1, -1, 0, 0, 0, 0, -1,
  0, -1, 0, 0, 0, 0, -1, 0, 0, 0,
  0, -1, 0, -1, 0, 0, 0, 0, -1, 0,
  0, 0, 0, -1, 0, -1, -1, 0, 0, 0,
  -1, 0, 0, 0, 0, 0, 0, -1, 2, 0,
  0, 0, -1, -1, 0, -1, 0, 0, 0, 0,
];

const elements = {
  grid: document.querySelector("#grid"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  modelType: document.querySelector("#model-type"),
  qValues: document.querySelector("#q-values"),
  previousAction: document.querySelector("#previous-action"),
  selectedAction: document.querySelector("#selected-action"),
  decisionNote: document.querySelector("#decision-note"),
  runCheck: document.querySelector("#run-check"),
  errorMessage: document.querySelector("#error-message"),
};

const actionSymbols = ["↑", "↓", "←", "→"];
let model = null;
let qValues = null;

function renderGrid() {
  const fragment = document.createDocumentFragment();
  OBSERVATION_GRID.forEach((value, index) => {
    const cell = document.createElement("span");
    cell.className = "grid-cell";
    cell.setAttribute("aria-label", `Row ${Math.floor(index / 10) + 1}, column ${(index % 10) + 1}`);

    if (value === -1) cell.classList.add("is-obstacle");
    if (value === 1) {
      cell.classList.add("is-agent");
      cell.textContent = "A";
    }
    if (value === 2) {
      cell.classList.add("is-goal");
      cell.textContent = "G";
    }
    fragment.append(cell);
  });
  elements.grid.replaceChildren(fragment);
}

function setStatus(state, message) {
  elements.statusDot.dataset.state = state;
  elements.statusText.textContent = message;
}

function showError(error) {
  elements.errorMessage.hidden = false;
  elements.errorMessage.textContent = error.message;
  setStatus("error", "Browser inference failed");
}

function renderDecision() {
  if (!model || !qValues) return;

  const selectedValue = elements.previousAction.value;
  const previousAction = selectedValue === "" ? null : Number(selectedValue);
  const decision = chooseAction(qValues, model.manifest, previousAction);

  elements.qValues.replaceChildren(
    ...qValues.map((value, index) => {
      const item = document.createElement("div");
      item.className = "q-value";
      if (index === decision.actionIndex) item.classList.add("is-selected");
      item.innerHTML = `
        <span>${actionSymbols[index]} ${model.manifest.output.actions[index]}</span>
        <strong>${value.toFixed(4)}</strong>
        <small>${decision.adjustedValues[index].toFixed(4)} adjusted</small>
      `;
      return item;
    }),
  );

  elements.selectedAction.textContent = `${actionSymbols[decision.actionIndex]} ${decision.actionName}`;
  elements.decisionNote.textContent = decision.inertiaApplied
    ? `Inertia applied with strength ${model.manifest.policy.inertia.strength}.`
    : "Raw model ranking; no inertia adjustment applied.";
}

async function runInference() {
  elements.runCheck.disabled = true;
  elements.errorMessage.hidden = true;
  setStatus("loading", "Loading ONNX Runtime and model");

  try {
    if (!model) {
      const manifestUrl = new URL(
        "models/frozen_transfer.json",
        document.baseURI,
      );
      model = await loadGridWorldModel(manifestUrl);
      elements.modelType.textContent = model.manifest.checkpoint_type;
    }

    qValues = await model.predict(OBSERVATION_GRID);
    renderDecision();
    setStatus("ready", "Browser inference succeeded");
  } catch (error) {
    showError(error instanceof Error ? error : new Error(String(error)));
  } finally {
    elements.runCheck.disabled = false;
  }
}

renderGrid();
elements.runCheck.addEventListener("click", runInference);
elements.previousAction.addEventListener("change", renderDecision);

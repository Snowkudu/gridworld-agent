import "./styles.css";

import { GridWorld } from "./gridworld.js";
import { loadGridWorldModel } from "./model-runtime.js";
import { chooseAction } from "./policy.js";
import {
  createSeededRandom,
  generateWorldObservation,
  WORLD_TIERS,
} from "./world-generator.js";

const DEMO_WORLD = [
  0, 0, -1, 0, 0, 0, -1, 0, -1, 0,
  0, 1, -1, 0, -1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, -1, 0, -1, -1, 0, -1,
  -1, -1, 0, 0, 0, 0, 0, -1, 0, 0,
  0, -1, 0, -1, -1, 0, 0, 0, 0, -1,
  0, -1, 0, 0, 0, 0, -1, 0, 0, 0,
  0, -1, 0, -1, 0, 0, 0, 0, -1, 0,
  0, 0, 0, -1, 0, -1, -1, 0, 0, 0,
  -1, 0, 0, 0, 0, 0, 0, -1, 2, 0,
  0, 0, -1, -1, 0, -1, 0, 0, -1, 0,
];

const elements = {
  grid: document.querySelector("#grid"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  exploreMode: document.querySelector("#explore-mode"),
  storyMode: document.querySelector("#story-mode"),
  exploreWorkspace: document.querySelector("#explore-workspace"),
  storyView: document.querySelector("#story-view"),
  storyVideo: document.querySelector("#story-video"),
  storyPosition: document.querySelector("#story-position"),
  storyTitle: document.querySelector("#story-title"),
  storyCopy: document.querySelector("#story-copy"),
  storyMetric: document.querySelector("#story-metric"),
  storySteps: [...document.querySelectorAll("[data-story-step]")],
  speedButtons: [...document.querySelectorAll("[data-speed]")],
  modelType: document.querySelector("#model-type"),
  qValues: document.querySelector("#q-values"),
  previousAction: document.querySelector("#previous-action"),
  selectedAction: document.querySelector("#selected-action"),
  stepCount: document.querySelector("#step-count"),
  lastEvent: document.querySelector("#last-event"),
  stepOnce: document.querySelector("#step-once"),
  runRollout: document.querySelector("#run-rollout"),
  resetInstance: document.querySelector("#reset-instance"),
  worldTools: document.querySelector("#world-tools"),
  randomWorld: document.querySelector("#random-world"),
  tierButtons: [...document.querySelectorAll("[data-tier]")],
  errorMessage: document.querySelector("#error-message"),
};

const actionSymbols = ["↑", "↓", "←", "→"];
const eventLabels = {
  reset: "Ready",
  moved: "Moved",
  illegal_move: "Boundary collision",
  obstacle_hit: "Obstacle collision",
  goal: "Goal reached",
  timeout: "Timed out",
};

const STORY_STAGES = [
  {
    title: "MLP — Open-loop success, closed-loop failure",
    copy: "It learned the oracle labels, but small mistakes compounded once its own actions created the next state.",
    metric: "45.1% test accuracy · 4% fully autonomous success",
  },
  {
    title: "CNN — Better spatial reasoning, weak recovery",
    copy: "Three spatial channels improved navigation, but the policy still struggled to backtrack or escape after making a mistake.",
    metric: "≈70% autonomous success · remaining failures were deadlocks and oscillations",
  },
  {
    title: "DDQN — A competent small-world controller",
    copy: "Learning from interaction produced a much stronger closed-loop policy without an oracle, rescue policy, or action mask.",
    metric: "Historical scratch champion · 88.4% on W0",
  },
  {
    title: "World tiers exposed long-horizon brittleness",
    copy: "As the minimum optimal route grew from W0 to W15, success fell sharply. The difficult run beside this text shows that limitation directly.",
    metric: "Frozen transfer · 87.2% W0 → 44.2% W15",
  },
  {
    title: "Frozen transfer revealed the real bottleneck",
    copy: "Frozen supervised CNN features nearly matched the scratch champion. Representation was not the main final limitation.",
    metric: "Scratch 88.4 / 42.0 · Frozen 87.2 / 44.2 · W0 / W15",
  },
];

function createWorld(observation) {
  return new GridWorld({
    observation,
    maxSteps: 50,
    expectedObstacleCount: 30,
  });
}

let world = createWorld(DEMO_WORLD);
let model = null;
let qValues = null;
let previousAction = null;
let rolloutId = 0;
let rolloutState = "idle";
let oneStepBusy = false;
let appMode = "explore";
let selectedTier = 0;
let rolloutDuration = 660;

function renderGrid(observation = world.observationGrid()) {
  let cells = [...elements.grid.querySelectorAll(".grid-cell")];
  if (cells.length !== observation.length) {
    for (const cell of cells) cell.remove();
    const fragment = document.createDocumentFragment();
    cells = observation.map((_, index) => {
      const cell = document.createElement("span");
      cell.className = "grid-cell";
      cell.setAttribute(
        "aria-label",
        `Row ${Math.floor(index / 10) + 1}, column ${(index % 10) + 1}`,
      );
      fragment.append(cell);
      return cell;
    });
    elements.grid.append(fragment);
  }

  observation.forEach((value, index) => {
    const cell = cells[index];
    cell.className = "grid-cell";
    cell.textContent = "";
    if (value === -1) cell.classList.add("is-obstacle");
    if (value === 1) {
      cell.classList.add("is-agent");
      cell.textContent = "A";
    }
    if (value === 2) {
      cell.classList.add("is-goal");
      cell.textContent = "G";
    }
  });
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

function renderDecision(decision = null) {
  if (!model || !qValues) return;

  const resolvedDecision =
    decision ?? chooseAction(qValues, model.manifest, previousAction);

  elements.qValues.replaceChildren(
    ...qValues.map((value, index) => {
      const item = document.createElement("div");
      item.className = "q-value";
      if (index === resolvedDecision.actionIndex) {
        item.classList.add("is-selected");
      }
      item.innerHTML = `
        <span>${actionSymbols[index]} ${model.manifest.output.actions[index]}</span>
        <strong>${value.toFixed(4)}</strong>
        <small>${resolvedDecision.adjustedValues[index].toFixed(4)} adjusted</small>
      `;
      return item;
    }),
  );

  elements.selectedAction.textContent = `${actionSymbols[resolvedDecision.actionIndex]} ${resolvedDecision.actionName}`;
}

function renderWorldStatus(event = world.lastEvent) {
  elements.stepCount.textContent = `Step ${world.currentSteps} / ${world.maxSteps}`;
  elements.lastEvent.textContent = eventLabels[event] ?? event;
}

function renderControls() {
  elements.stepOnce.disabled = oneStepBusy || rolloutState === "running";
  elements.runRollout.disabled = oneStepBusy;

  const rolloutLabels = {
    idle: "Run policy",
    running: "Pause",
    paused: "Resume",
    complete: "Replay world",
  };
  elements.runRollout.textContent = rolloutLabels[rolloutState];
}

function renderSpeedControl() {
  for (const button of elements.speedButtons) {
    const selected = Number(button.dataset.speed) === rolloutDuration;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function renderMode() {
  const exploring = appMode === "explore";
  elements.exploreMode.classList.toggle("is-active", exploring);
  elements.storyMode.classList.toggle("is-active", !exploring);
  elements.exploreMode.setAttribute("aria-pressed", String(exploring));
  elements.storyMode.setAttribute("aria-pressed", String(!exploring));
  elements.exploreWorkspace.hidden = !exploring;
  elements.storyView.hidden = exploring;
  elements.worldTools.hidden = !exploring;
  renderControls();
}

function renderStoryStage(stageIndex) {
  const stage = STORY_STAGES[stageIndex];
  elements.storyPosition.textContent = `${String(stageIndex + 1).padStart(2, "0")} / 05`;
  elements.storyTitle.textContent = stage.title;
  elements.storyCopy.textContent = stage.copy;
  elements.storyMetric.textContent = stage.metric;
  for (const button of elements.storySteps) {
    const selected = Number(button.dataset.storyStep) === stageIndex;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function renderTierSelection() {
  for (const button of elements.tierButtons) {
    const selected = Number(button.dataset.tier) === selectedTier;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function selectMode(nextMode) {
  if (nextMode === appMode) return;

  rolloutId += 1;
  appMode = nextMode;

  if (nextMode === "explore") {
    elements.storyVideo.pause();
    setStatus(
      "ready",
      world.currentSteps > 0
        ? `Explore mode · state preserved at step ${world.currentSteps}`
        : "Explore mode ready",
    );
    renderMode();
    return;
  }

  if (rolloutState === "running") rolloutState = "paused";
  elements.storyVideo.currentTime = 0;
  void elements.storyVideo.play().catch(() => {});
  setStatus("ready", "Project story · final DQN run shown beside the evidence");
  renderMode();
}

function syncPreviousActionControl() {
  elements.previousAction.textContent =
    previousAction === null
      ? "None"
      : `${actionSymbols[previousAction]} ${model?.manifest.output.actions[previousAction] ?? ""}`;
}

async function ensureModel() {
  if (model) return model;

  setStatus("loading", "Loading ONNX Runtime and model");
  const manifestUrl = new URL(
    "models/frozen_transfer.json",
    document.baseURI,
  );
  model = await loadGridWorldModel(manifestUrl);
  elements.modelType.textContent = model.manifest.checkpoint_type;
  setStatus("ready", "Browser model ready");
  return model;
}

async function takePolicyStep({
  animationDelay = 0,
  expectedRolloutId = null,
} = {}) {
  if (world.done) return null;

  await ensureModel();
  qValues = await model.predict(world.observationGrid());
  if (expectedRolloutId !== null && expectedRolloutId !== rolloutId) return null;

  const decision = chooseAction(qValues, model.manifest, previousAction);
  renderDecision(decision);

  if (animationDelay > 0) {
    await new Promise((resolve) => window.setTimeout(resolve, animationDelay));
  }
  if (expectedRolloutId !== null && expectedRolloutId !== rolloutId) return null;

  const result = world.step(decision.actionIndex);
  if (result.moved) {
    previousAction = decision.actionIndex;
    syncPreviousActionControl();
  }

  renderGrid(result.observation);
  renderWorldStatus(result.event);
  return result;
}

async function stepOnce() {
  const currentRolloutId = ++rolloutId;
  oneStepBusy = true;
  renderControls();
  elements.errorMessage.hidden = true;

  try {
    if (world.done) {
      setStatus("ready", "Reset the completed world to continue");
      return;
    }

    const result = await takePolicyStep({ expectedRolloutId: currentRolloutId });
    if (currentRolloutId !== rolloutId || !result) return;
    if (result.done) rolloutState = "complete";

    setStatus(
      "ready",
      result.done
        ? result.success
          ? `Goal reached in ${result.currentSteps} steps`
          : `Rollout timed out after ${result.currentSteps} steps`
        : `Policy step ${result.currentSteps} completed`,
    );
  } catch (error) {
    showError(error instanceof Error ? error : new Error(String(error)));
  } finally {
    if (currentRolloutId === rolloutId) {
      oneStepBusy = false;
      renderControls();
    }
  }
}

async function runAutonomousRollout({ reset = true } = {}) {
  const currentRolloutId = ++rolloutId;
  if (reset) {
    world.reset();
    previousAction = null;
    qValues = null;
    syncPreviousActionControl();
    renderGrid();
    renderWorldStatus();
  }
  rolloutState = "running";
  renderControls();
  elements.errorMessage.hidden = true;

  try {
    await ensureModel();

    while (!world.done && currentRolloutId === rolloutId) {
      const stepDuration = rolloutDuration;
      const result = await takePolicyStep({
        animationDelay: Math.round(stepDuration * 0.68),
        expectedRolloutId: currentRolloutId,
      });
      if (!result) break;
      setStatus("loading", `Autonomous rollout · step ${result.currentSteps}`);
      await new Promise((resolve) =>
        window.setTimeout(resolve, Math.round(stepDuration * 0.32)),
      );
    }

    if (currentRolloutId !== rolloutId) return;
    rolloutState = "complete";
    setStatus(
      "ready",
      world.lastEvent === "goal"
        ? `Goal reached in ${world.currentSteps} steps`
        : `Rollout timed out after ${world.currentSteps} steps`,
    );
  } catch (error) {
    rolloutState = "idle";
    showError(error instanceof Error ? error : new Error(String(error)));
  } finally {
    if (currentRolloutId === rolloutId) renderControls();
  }
}

function toggleRollout() {
  if (rolloutState === "running") {
    rolloutId += 1;
    rolloutState = "paused";
    renderControls();
    setStatus("ready", `Policy paused at step ${world.currentSteps}`);
    return;
  }

  void runAutonomousRollout({ reset: rolloutState !== "paused" });
}

function freshSeed() {
  const values = new Uint32Array(1);
  window.crypto.getRandomValues(values);
  return values[0];
}

function resetInstance() {
  rolloutId += 1;
  rolloutState = "idle";
  oneStepBusy = false;
  world.reset();
  previousAction = null;
  qValues = null;
  syncPreviousActionControl();
  renderGrid();
  renderWorldStatus();
  elements.qValues.replaceChildren();
  elements.selectedAction.textContent = "—";
  elements.errorMessage.hidden = true;
  renderControls();
  setStatus("ready", "Current world reset to its starting state");
}

function randomWorld() {
  rolloutId += 1;
  rolloutState = "idle";
  oneStepBusy = false;
  const tier = WORLD_TIERS[selectedTier];
  const generated = generateWorldObservation({
    minSolutionSteps: tier.minSolutionSteps,
    rng: createSeededRandom(freshSeed()),
  });
  world = createWorld(generated.observation);
  previousAction = null;
  qValues = null;
  syncPreviousActionControl();
  renderGrid();
  renderWorldStatus();
  elements.qValues.replaceChildren();
  elements.selectedAction.textContent = "—";
  elements.errorMessage.hidden = true;
  renderControls();
  setStatus(
    "ready",
    `${tier.label} W${selectedTier} world ready · optimal path ${generated.solutionSteps} steps`,
  );
}

renderGrid();
renderWorldStatus();
renderTierSelection();
renderSpeedControl();
renderStoryStage(0);
renderMode();
elements.stepOnce.addEventListener("click", stepOnce);
elements.runRollout.addEventListener("click", toggleRollout);
elements.resetInstance.addEventListener("click", resetInstance);
elements.randomWorld.addEventListener("click", randomWorld);
for (const button of elements.tierButtons) {
  button.addEventListener("click", () => {
    selectedTier = Number(button.dataset.tier);
    renderTierSelection();
    randomWorld();
  });
}
elements.exploreMode.addEventListener("click", () => selectMode("explore"));
elements.storyMode.addEventListener("click", () => selectMode("story"));
for (const button of elements.storySteps) {
  button.addEventListener("click", () =>
    renderStoryStage(Number(button.dataset.storyStep)),
  );
}
for (const button of elements.speedButtons) {
  button.addEventListener("click", () => {
    rolloutDuration = Number(button.dataset.speed);
    renderSpeedControl();
  });
}

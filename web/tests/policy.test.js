import assert from "node:assert/strict";
import test from "node:test";

import { chooseAction } from "../src/policy.js";

const manifest = {
  output: {
    actions: ["up", "down", "left", "right"],
  },
  policy: {
    inertia: {
      enabled: true,
      strength: 0.75,
    },
  },
};

test("selects the raw argmax without a previous action", () => {
  const decision = chooseAction([0.1, 0.8, 0.2, 0.3], manifest);

  assert.equal(decision.actionIndex, 1);
  assert.equal(decision.actionName, "down");
  assert.equal(decision.inertiaApplied, false);
});

test("penalizes the action opposite to the previous action", () => {
  const decision = chooseAction([0.1, 0.8, 0.2, 0.3], manifest, 0);

  assert.ok(Math.abs(decision.adjustedValues[1] - 0.05) < 1e-12);
  assert.equal(decision.actionIndex, 3);
  assert.equal(decision.actionName, "right");
  assert.equal(decision.inertiaApplied, true);
});

test("preserves first-action tie breaking", () => {
  const decision = chooseAction([1, 1, 1, 1], manifest);

  assert.equal(decision.actionIndex, 0);
});

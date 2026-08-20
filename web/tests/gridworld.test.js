import assert from "node:assert/strict";
import test from "node:test";

import { AGENT, GOAL, GridWorld, OBSTACLE } from "../src/gridworld.js";

function makeObservation({ agent = [1, 1], goal = [2, 2], obstacles = [] } = {}) {
  const size = 10;
  const observation = Array(size * size).fill(0);
  for (const [row, column] of obstacles) {
    observation[row * size + column] = OBSTACLE;
  }
  observation[agent[0] * size + agent[1]] = AGENT;
  observation[goal[0] * size + goal[1]] = GOAL;
  return observation;
}

test("moves in all four action directions", () => {
  const cases = [
    [0, [4, 5]],
    [1, [6, 5]],
    [2, [5, 4]],
    [3, [5, 6]],
  ];

  for (const [action, expectedState] of cases) {
    const world = new GridWorld({
      observation: makeObservation({ agent: [5, 5], goal: [9, 9] }),
    });
    const result = world.step(action);

    assert.deepEqual(result.state, expectedState);
    assert.equal(result.event, "moved");
    assert.equal(result.moved, true);
  }
});

test("boundary collision preserves the current state", () => {
  const world = new GridWorld({
    observation: makeObservation({ agent: [0, 0], goal: [9, 9] }),
  });

  const result = world.step(0);

  assert.deepEqual(result.state, [0, 0]);
  assert.equal(result.event, "illegal_move");
  assert.equal(result.moved, false);
});

test("obstacle collision preserves the current state", () => {
  const world = new GridWorld({
    observation: makeObservation({
      agent: [1, 1],
      goal: [9, 9],
      obstacles: [[1, 2]],
    }),
  });

  const result = world.step(3);

  assert.deepEqual(result.state, [1, 1]);
  assert.equal(result.event, "obstacle_hit");
  assert.equal(result.moved, false);
});

test("reaching the goal completes the world", () => {
  const world = new GridWorld({
    observation: makeObservation({ agent: [1, 1], goal: [1, 2] }),
  });

  const result = world.step(3);

  assert.equal(result.event, "goal");
  assert.equal(result.done, true);
  assert.equal(result.success, true);
  assert.equal(result.observation.filter((value) => value === AGENT).length, 1);
});

test("maxSteps converts the final non-goal event into timeout", () => {
  const world = new GridWorld({
    observation: makeObservation({ agent: [1, 1], goal: [9, 9] }),
    maxSteps: 1,
  });

  const result = world.step(1);

  assert.equal(result.event, "timeout");
  assert.equal(result.done, true);
  assert.equal(result.success, false);
});

test("reset restores the initial contract", () => {
  const world = new GridWorld({
    observation: makeObservation({ agent: [1, 1], goal: [9, 9] }),
  });
  world.step(1);

  const observation = world.reset();

  assert.deepEqual(world.state, [1, 1]);
  assert.equal(world.currentSteps, 0);
  assert.equal(world.done, false);
  assert.equal(observation.filter((value) => value === AGENT).length, 1);
  assert.equal(observation.filter((value) => value === GOAL).length, 1);
});

test("can enforce the canonical obstacle count", () => {
  assert.throws(
    () =>
      new GridWorld({
        observation: makeObservation(),
        expectedObstacleCount: 30,
      }),
    /Expected 30 obstacles, found 0/,
  );
});

test("shortestPath follows a valid route around obstacles", () => {
  const world = new GridWorld({
    observation: makeObservation({
      agent: [1, 1],
      goal: [1, 3],
      obstacles: [[1, 2]],
    }),
  });

  const path = world.shortestPath();

  assert.deepEqual(path[0], [1, 1]);
  assert.deepEqual(path.at(-1), [1, 3]);
  assert.equal(path.length - 1, 4);
  assert.equal(path.some(([row, column]) => row === 1 && column === 2), false);
});

test("shortestPath starts from the agent's current state", () => {
  const world = new GridWorld({
    observation: makeObservation({ agent: [1, 1], goal: [3, 1] }),
  });
  world.step(1);

  assert.deepEqual(world.shortestPath(), [
    [2, 1],
    [3, 1],
  ]);
});

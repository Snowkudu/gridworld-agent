import assert from "node:assert/strict";
import test from "node:test";

import { GridWorld, OBSTACLE } from "../src/gridworld.js";
import {
  createSeededRandom,
  generateWorldObservation,
} from "../src/world-generator.js";

for (const minimumSteps of [0, 5, 10, 15]) {
  test(`generates a solvable W${minimumSteps} world`, () => {
    const generated = generateWorldObservation({
      minSolutionSteps: minimumSteps,
      rng: createSeededRandom(1000 + minimumSteps),
    });
    const world = new GridWorld({
      observation: generated.observation,
      expectedObstacleCount: 30,
    });

    assert.equal(
      generated.observation.filter((cell) => cell === OBSTACLE).length,
      30,
    );
    assert.ok(world.shortestPath());
    assert.ok(generated.solutionSteps >= minimumSteps);
  });
}

test("seeded generation is reproducible", () => {
  const first = generateWorldObservation({ rng: createSeededRandom(42) });
  const second = generateWorldObservation({ rng: createSeededRandom(42) });

  assert.deepEqual(first, second);
});

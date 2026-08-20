import { AGENT, EMPTY, GOAL, GridWorld, OBSTACLE } from "./gridworld.js";

export const WORLD_TIERS = Object.freeze({
  0: { label: "Easy", minSolutionSteps: 0 },
  5: { label: "Medium", minSolutionSteps: 5 },
  10: { label: "Hard", minSolutionSteps: 10 },
  15: { label: "Very hard", minSolutionSteps: 15 },
});

export function createSeededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function randomIndex(rng, length) {
  return Math.floor(rng() * length);
}

export function generateWorldObservation({
  minSolutionSteps = 0,
  size = 10,
  obstacleCount = 30,
  maxAttempts = 10_000,
  rng = Math.random,
} = {}) {
  if (!Number.isInteger(minSolutionSteps) || minSolutionSteps < 0) {
    throw new Error("minSolutionSteps must be a non-negative integer.");
  }
  if (obstacleCount < 0 || obstacleCount > size * size - 2) {
    throw new Error("obstacleCount must leave room for the agent and goal.");
  }

  const cellCount = size * size;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const available = Array.from({ length: cellCount }, (_, index) => index);
    const takeCell = () => {
      const selection = randomIndex(rng, available.length);
      return available.splice(selection, 1)[0];
    };

    const agentIndex = takeCell();
    const goalIndex = takeCell();
    const obstacleIndices = Array.from({ length: obstacleCount }, takeCell);
    const observation = Array(cellCount).fill(EMPTY);
    observation[agentIndex] = AGENT;
    observation[goalIndex] = GOAL;
    for (const index of obstacleIndices) observation[index] = OBSTACLE;

    const candidate = new GridWorld({
      observation,
      size,
      expectedObstacleCount: obstacleCount,
    });
    const shortestPath = candidate.shortestPath();
    const solutionSteps = shortestPath === null ? null : shortestPath.length - 1;

    if (solutionSteps !== null && solutionSteps >= minSolutionSteps) {
      return { observation, solutionSteps };
    }
  }

  throw new Error(
    `Could not generate a solvable world with a minimum path of ${minSolutionSteps} steps.`,
  );
}

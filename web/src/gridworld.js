export const EMPTY = 0;
export const AGENT = 1;
export const GOAL = 2;
export const OBSTACLE = -1;

export const ACTION_DELTAS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

function samePosition(left, right) {
  return left[0] === right[0] && left[1] === right[1];
}

function positionFromIndex(index, size) {
  return [Math.floor(index / size), index % size];
}

function indexFromPosition([row, column], size) {
  return row * size + column;
}

export class GridWorld {
  constructor({
    observation,
    maxSteps = 100,
    size = 10,
    expectedObstacleCount = null,
  }) {
    if (!Array.isArray(observation) || observation.length !== size * size) {
      throw new Error(`Expected a flattened ${size} × ${size} observation.`);
    }
    if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
      throw new Error("maxSteps must be a positive integer.");
    }

    const allowedValues = new Set([OBSTACLE, EMPTY, AGENT, GOAL]);
    if (observation.some((value) => !allowedValues.has(value))) {
      throw new Error("Observation contains an unsupported cell value.");
    }

    const obstacleCount = observation.filter(
      (value) => value === OBSTACLE,
    ).length;
    if (
      expectedObstacleCount !== null &&
      obstacleCount !== expectedObstacleCount
    ) {
      throw new Error(
        `Expected ${expectedObstacleCount} obstacles, found ${obstacleCount}.`,
      );
    }

    const agentIndices = [];
    const goalIndices = [];
    observation.forEach((value, index) => {
      if (value === AGENT) agentIndices.push(index);
      if (value === GOAL) goalIndices.push(index);
    });

    if (agentIndices.length !== 1 || goalIndices.length !== 1) {
      throw new Error("Observation must contain exactly one agent and one goal.");
    }

    this.size = size;
    this.maxSteps = maxSteps;
    this.obstacleCount = obstacleCount;
    this.startState = positionFromIndex(agentIndices[0], size);
    this.goalState = positionFromIndex(goalIndices[0], size);
    this.baseGrid = observation.map((value) =>
      value === OBSTACLE ? OBSTACLE : EMPTY,
    );

    this.reset();
  }

  reset() {
    this.state = [...this.startState];
    this.currentSteps = 0;
    this.done = false;
    this.lastEvent = "reset";
    return this.observationGrid();
  }

  observationGrid() {
    const observation = [...this.baseGrid];
    observation[indexFromPosition(this.goalState, this.size)] = GOAL;
    observation[indexFromPosition(this.state, this.size)] = AGENT;
    return observation;
  }

  isLegal(action) {
    this.#validateAction(action);
    const candidate = this.#candidatePosition(action);
    return this.#isInBounds(candidate) && !this.#isObstacle(candidate);
  }

  shortestPath() {
    const startIndex = indexFromPosition(this.state, this.size);
    const goalIndex = indexFromPosition(this.goalState, this.size);
    const queue = [startIndex];
    const parents = new Map([[startIndex, null]]);

    for (let head = 0; head < queue.length; head += 1) {
      const currentIndex = queue[head];
      if (currentIndex === goalIndex) break;

      const [row, column] = positionFromIndex(currentIndex, this.size);
      for (const [rowDelta, columnDelta] of ACTION_DELTAS) {
        const candidate = [row + rowDelta, column + columnDelta];
        if (!this.#isInBounds(candidate) || this.#isObstacle(candidate)) continue;

        const candidateIndex = indexFromPosition(candidate, this.size);
        if (parents.has(candidateIndex)) continue;
        parents.set(candidateIndex, currentIndex);
        queue.push(candidateIndex);
      }
    }

    if (!parents.has(goalIndex)) return null;

    const path = [];
    let currentIndex = goalIndex;
    while (currentIndex !== null) {
      path.push(positionFromIndex(currentIndex, this.size));
      currentIndex = parents.get(currentIndex);
    }
    return path.reverse();
  }

  step(action) {
    this.#validateAction(action);
    if (this.done) {
      throw new Error("Cannot step a completed GridWorld. Reset it first.");
    }

    const previousState = [...this.state];
    const candidate = this.#candidatePosition(action);
    this.currentSteps += 1;

    let event;
    if (!this.#isInBounds(candidate)) {
      event = "illegal_move";
    } else if (this.#isObstacle(candidate)) {
      event = "obstacle_hit";
    } else {
      this.state = candidate;
      event = samePosition(this.state, this.goalState) ? "goal" : "moved";
    }

    if (event === "goal") {
      this.done = true;
    } else if (this.currentSteps >= this.maxSteps) {
      event = "timeout";
      this.done = true;
    }

    this.lastEvent = event;

    return {
      observation: this.observationGrid(),
      previousState,
      state: [...this.state],
      event,
      done: this.done,
      moved: !samePosition(previousState, this.state),
      success: event === "goal",
      currentSteps: this.currentSteps,
    };
  }

  #validateAction(action) {
    if (!Number.isInteger(action) || action < 0 || action >= ACTION_DELTAS.length) {
      throw new Error(`Unknown action: ${action}`);
    }
  }

  #candidatePosition(action) {
    const [rowDelta, columnDelta] = ACTION_DELTAS[action];
    return [this.state[0] + rowDelta, this.state[1] + columnDelta];
  }

  #isInBounds([row, column]) {
    return row >= 0 && row < this.size && column >= 0 && column < this.size;
  }

  #isObstacle(position) {
    return this.baseGrid[indexFromPosition(position, this.size)] === OBSTACLE;
  }
}

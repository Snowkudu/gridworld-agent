const OPPOSITE_ACTION = [1, 0, 3, 2];

export function chooseAction(qValues, manifest, previousAction = null) {
  if (!Array.isArray(qValues) || qValues.length !== 4) {
    throw new Error("Expected four Q-values.");
  }

  const adjustedValues = [...qValues];
  const inertia = manifest.policy?.inertia;

  if (
    inertia?.enabled &&
    previousAction !== null &&
    Number.isInteger(previousAction)
  ) {
    adjustedValues[OPPOSITE_ACTION[previousAction]] -= inertia.strength;
  }

  const actionIndex = adjustedValues.reduce(
    (bestIndex, value, index, values) =>
      value > values[bestIndex] ? index : bestIndex,
    0,
  );

  return {
    actionIndex,
    actionName: manifest.output.actions[actionIndex],
    adjustedValues,
    inertiaApplied: adjustedValues.some(
      (value, index) => value !== qValues[index],
    ),
  };
}

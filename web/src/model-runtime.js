import * as ort from "onnxruntime-web/wasm";

const EXPECTED_CHANNELS = ["obstacles", "agent", "goal"];
const EXPECTED_ACTIONS = ["up", "down", "left", "right"];

function arraysEqual(left, right) {
  return (
    Array.isArray(left) &&
    Array.isArray(right) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function validateManifest(manifest) {
  if (manifest.schema_version !== 1) {
    throw new Error(`Unsupported model schema: ${manifest.schema_version}`);
  }
  if (!manifest.model_file) {
    throw new Error("Model manifest does not specify model_file.");
  }
  if (!arraysEqual(manifest.input?.shape, [1, 3, 10, 10])) {
    throw new Error("Expected model input shape [1, 3, 10, 10].");
  }
  if (manifest.input?.dtype !== "float32") {
    throw new Error("Expected a float32 model input.");
  }
  if (!arraysEqual(manifest.input?.channels, EXPECTED_CHANNELS)) {
    throw new Error("Model channel order does not match the browser encoder.");
  }
  if (!arraysEqual(manifest.output?.actions, EXPECTED_ACTIONS)) {
    throw new Error("Model action order does not match the browser policy.");
  }
}

function encodeGrid(grid) {
  if (!Array.isArray(grid) || grid.length !== 100) {
    throw new Error("Expected a flattened 10 × 10 observation grid.");
  }

  const agentCount = grid.filter((value) => value === 1).length;
  const goalCount = grid.filter((value) => value === 2).length;
  if (agentCount !== 1 || goalCount !== 1) {
    throw new Error("Observation grid must contain exactly one agent and one goal.");
  }

  const tensorData = new Float32Array(300);
  for (let index = 0; index < grid.length; index += 1) {
    tensorData[index] = grid[index] === -1 ? 1 : 0;
    tensorData[100 + index] = grid[index] === 1 ? 1 : 0;
    tensorData[200 + index] = grid[index] === 2 ? 1 : 0;
  }

  return new ort.Tensor("float32", tensorData, [1, 3, 10, 10]);
}

export async function loadGridWorldModel(manifestUrl) {
  const response = await fetch(manifestUrl, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(
      `Could not load model manifest (${response.status} ${response.statusText}).`,
    );
  }

  const manifest = await response.json();
  validateManifest(manifest);

  const modelUrl = new URL(manifest.model_file, manifestUrl);
  const session = await ort.InferenceSession.create(modelUrl.href, {
    executionProviders: ["wasm"],
  });

  return {
    manifest,
    async predict(grid) {
      const input = encodeGrid(grid);
      const results = await session.run({ [manifest.input.name]: input });
      const output = results[manifest.output.name];

      if (!output || output.data.length !== 4) {
        throw new Error("Model output does not contain four action values.");
      }

      return Array.from(output.data, Number);
    },
  };
}

import numpy as np
from utils.paths import raw_data_dir

ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

def verify_dataset(path):
    
    d = np.load(path, allow_pickle=True)
    print("File:", path)
    print("Keys:", list(d.keys()))

    states = d["states"]
    actions = d["actions"]

    # 1) shapes + dtypes
    print("states:", states.shape, states.dtype)
    print("actions:", actions.shape, actions.dtype)

    # 2) encoding sanity  checks for the content of states.
    uvals = np.unique(states)
    print("state min/max:", float(states.min()), float(states.max()))
    print("unique state values:", uvals)

    # 3) action sanity action checks for the amount of actions.
    print("actions min/max:", int(actions.min()), int(actions.max()))
    counts = np.bincount(actions.astype(np.int64), minlength=4)
    print("action dist:", counts.tolist(), "pct:", (counts / counts.sum()).round(4).tolist())
    # 4) invariants on a few samples
    size = 10
    for idx in np.random.choice(states.shape[0], 5, replace=False):
        g = states[idx].reshape(size, size)
        agent_count = int((g == 1).sum())
        goal_count = int((g == 2).sum())
        obs_count = int((g == -1).sum())
        if agent_count != 1 or goal_count != 1:
            print(f"[FAIL] sample {idx}: agent={agent_count}, goal={goal_count}, obstacles={obs_count}")
            return
       

        # 5) legal action rate (single line)
    ok = 0
    trials = 200
    for idx in np.random.choice(states.shape[0], trials, replace=False):
        g = states[idx].reshape(size, size)
        r, c = np.argwhere(g == 1)[0]
        a = int(actions[idx])
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size and g[nr, nc] != -1:
            ok += 1
    print(f"Legal action rate: {ok}/{trials} ({ok/trials:.3f})")
    print("Invariant check: OK (5 random samples)")
    print(">>> verification complete.")


def main():
    default_dataset_path = raw_data_dir() / "gridworld_2000ep_200ms_123seed.npz"
    verify_dataset(default_dataset_path)

if __name__ == "__main__":
    main()

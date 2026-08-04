from environment.gridworld import GridWorld
from environment.rewards import manhattan_shaped_reward
from policies.actions import ACTION_NAMES
from scripts.verify import validate_arrays  
from policies.oracle import OraclePolicy
import numpy as np
import os,time,argparse
print(">>> dataset.py loaded")


def generate_dataset(
    env:GridWorld,
    episodes:int,
    out_path:str,
    seed: int | None = None,
    overwrite: bool = False,
    verbose: bool = True,
   
):
   
    
    states=[]
    actions=[]
    labels=[]
    solved=0
    timed_out=0


    data = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_data = []
        while not done :
            oracleAction = OraclePolicy.select_action(env) # bfs policy to get the best action
            action = OraclePolicy.select_action(env) # will be random at one point, but for now we can use the oracle to generate the dataset
            actions.append(action)
        
            state = env.state_vector() # Flatten the grid to a 1D array and convert to float32
            states.append(state)
            labels.append(action)
            next_state, reward, done = env.step(action)
            
            episode_data.append((state, action, reward, next_state, done))
            
            if done and env.state == env.goal_state:
                solved += 1
            if done and env.state != env.goal_state:
                timed_out += 1

        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {episode+1}/{episodes} completed.")
    
    states=np.stack(states).astype(np.float32) #should be a format of (N, state_dim)
    actions=np.array(actions,dtype=np.int64)   #should be a format of (N,)

    meta ={
        "dataset_version": "gridworld_dataset_v1",
        "environment_version": "gridworld_env_v1",
        "reward_version": "manhattan_shaped_v1",
        "grid_size": getattr(env, "grid_size", "-1"), 
        "obstacle_density": float(env.obstaclesPercent) ,
        "obstacle_count": int(env.countobstacles),
         
        "episodes":episodes,
        "max_steps":env.maxsteps,
        "seed": seed  if seed is not None else "None",
        "action_order": tuple(ACTION_NAMES),
       
        "samples": int(states.shape[0]),
        "solved_episodes": int(sovled),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
   
   
    #make the file to save the outputs and compress it with some metadata aswell.
  
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not overwrite and os.path.exists(out_path):
     raise FileExistsError(
         f"Dataset already exists: {out_path}\n"
        "Pass --overwrite to replace it."
    )
    np.savez_compressed(out_path, states=states, actions=actions, meta=meta)

    validate_arrays(states, actions)  # Verify the dataset after saving
    print(f"Dataset saved to: {out_path}")
    print(f"Samples: {states.shape[0]} | Solved episodes: {sovled}/{episodes}")
    print(f"X shape: {states.shape} | y shape: {actions.shape}")
    return states, actions, meta
   
def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--episodes", type=int, default=2000)
     parser.add_argument("--max_steps", type=int, default=200)
     parser.add_argument("--seed", type=int, default=123)
     parser.add_argument("--outpath", type=str, default="")
     parser.add_argument("--overwrite", action="store_true")
     parser.add_argument("--verbose", action="store_true")
     
     args= parser.parse_args()
     print("Generating dataset..:")
     #init env
     env=GridWorld(10,0.3,manhattan_shaped_reward,maxsteps=args.max_steps,seed=args.seed)

     if not args.outpath:
         out_path = f"data/raw/gridworld_{args.episodes}ep_{args.max_steps}ms_{args.seed}seed.npz"
     x,y,meta=generate_dataset(
        env=env,
        episodes=args.episodes,
        out_path=args.outpath if args.outpath else out_path,
        seed=args.seed,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )
     print("Done.\n")
if __name__ == "__main__":
    main()
 


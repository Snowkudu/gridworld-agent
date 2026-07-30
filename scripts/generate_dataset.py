from html import parser
from environment.gridworld import GridWorld
import numpy as np
import os,time,argparse
print(">>> dataset.py loaded")


def generate_dataset(
    env:GridWorld,
    episodes:int,
    out_path:str,
    seed: int | None = None,
    verbose: bool = True,
):
    if seed is not None:
        np.random.seed(seed)
    
    states=[]
    actions=[]
    sovled=0


    data = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_data = []
        while not done :
            
            action = env.next_Action() # Random action (0-3)
            actions.append(action)
        
            state = env.grid.flatten().astype(np.float32) # Flatten the grid to a 1D array and convert to float32
            states.append(state)
            next_state, reward, done = env.step(action)
            
            episode_data.append((state, action, reward, next_state, done))

            if(done == True):
                sovled+=1
        

        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {episode+1}/{episodes} completed.")
    
    states=np.stack(states).astype(np.float32) #should be a format of (N, state_dim)
    actions=np.array(actions,dtype=np.int64)   #should be a format of (N,)

    meta ={
        "episodes":episodes,
        "max_steps":env.maxsteps,
        "seed": seed  if seed is not None else "None",
        "gridSize": getattr(env, "grid_size", "-1"), 
        "density":  getattr(env, "obstacle_density", "-1.0"),
        "samples": int(states.shape[0]),
        "solved": int(sovled),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
   
   
    #make the file to save the outputs and compress it with some metadata aswell.
  
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, states=states, actions=actions, meta=meta)
   
    print(f"Saved dataset: {out_path}")
    print(f"Samples: {states.shape[0]} | Solved episodes: {sovled}/{episodes}")
    print(f"X shape: {states.shape} | y shape: {actions.shape}")
    return states, actions, meta
   
def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--episodes", type=int, default=2000)
     parser.add_argument("--max_steps", type=int, default=200)
     parser.add_argument("--seed", type=int, default=123)
     parser.add_argument("--outpath", type=str, default="")
     parser.add_argument("--verbose", action="store_true")
     args= parser.parse_args()
     print("Generating dataset..:")
     #init env
     env=GridWorld(10,0.3)

     if not args.outpath:
         out_path = f"data/raw/gridworld_{args.episodes}ep_{args.max_steps}ms_{args.seed}seed.npz"
     x,y,meta=generate_dataset(
        env=env,
        episodes=args.episodes,
        out_path=args.outpath if args.outpath else out_path,
        seed=args.seed,
        verbose=True
    )
     print("Done.\n")
if __name__ == "__main__":
    main()
 


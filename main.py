import random
from environment.gridworld import GridWorld
from environment.renderer import render
from environment.rewards import manhattan_shaped_reward
size=10

env = GridWorld(size, 0.3,manhattan_shaped_reward,50)

state = env.reset()
done = False

def read_action():
    try:
        import msvcrt
    except Exception:
        # Fallback to numeric input if msvcrt is not available
        while True:
            s = input("Move (0-3): ")
            if s in ("0", "1", "2", "3"):
                return int(s)
            print("Invalid input. Enter 0-3.")

    print("Press an arrow key to move (Up/Down/Left/Right), or 'q' to quit.")
    while True:
        ch = msvcrt.getch()
        # Special keys return a prefix (b"\x00" or b"\xe0") then a code
        if ch in (b"\x00", b"\xe0"):
            code = msvcrt.getch()
            if code == b'H':
                return 0  # Up
            if code == b'P':
                return 1  # Down
            if code == b'K':
                return 2  # Left
            if code == b'M':
                return 3  # Right
          
        else:
            if ch in (b'q', b'Q'):
                return None


while not done:
    render(env)
    action = read_action()
    if action is None:
        print("Exiting.")
        break
    state, reward, done = env.step(action)
    print("Reward:", reward,"\n")
    s = env.debug_tensor()

   

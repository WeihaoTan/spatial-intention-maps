import math
from env_pri import VectorEnv
import argparse
import pdb
import numpy as np
import pybullet as p
def main(args):
    #                   0              1           2            3
    #ACTIONLIST = ["move forward", "turn left", "turn right", "stay"]

    # Create env
    env = VectorEnv(use_egl_renderer=True, show_gui=True, room_length=1.0, room_width=0.5, obs_radius = 0.2, termination_step = 2000)
    

    # Run policy
    step = 0
    state = env.reset()
    print("step", step)
    print("state", state)
    print("red robot:")
    print("red robot position", state[0][0], state[0][1])
    print("red robot heading", state[0][2])
    print("green robot position", state[0][3], state[0][4])
    print("green robot heading", state[0][5])
    print("target position", state[0][6], state[0][7])
    print()
    print("green robot:")
    print("red robot position", state[1][0], state[1][1])
    print("red robot heading", state[1][2])
    print("green robot position", state[1][3], state[1][4])
    print("green robot heading", state[1][5])
    print("target position", state[1][6], state[1][7])    
    print("--------------------------------------------------------------------")
    
    
    print("state_size", env.state_size)
    print("obs_size", env.obs_size)
    print("n_action", env.n_action)
    print("action_spaces", env.action_spaces)
    print("get_avail_actions", env.get_avail_actions())
    print("action_space_sample", env.action_space_sample(0), env.action_space_sample(1)) 

    #pdb.set_trace()
    
    while True:
        #a = input("input:").split(" ")
        #action = [int(a[0]), int(a[1])]

        action = [env.action_space_sample(0), env.action_space_sample(1)]
        
        state, reward, done, _ = env.step(action)
        
        print(state)
        print("step", step)
        print("action", action)
        print("state", state)
        print("red robot:")
        print("red robot position", state[0][0], state[0][1])
        print("red robot heading", state[0][2])
        print("green robot position", state[0][3], state[0][4])
        print("green robot heading", state[0][5])
        print("target position", state[0][6], state[0][7])
        print()
        print("green robot:")
        print("red robot position", state[1][0], state[1][1])
        print("red robot heading", state[1][2])
        print("green robot position", state[1][3], state[1][4])
        print("green robot heading", state[1][5])
        print("target position", state[1][6], state[1][7])    
        print("reward", reward)
        print("--------------------------------------------------------------------")

        step += 1
        if done:
            break
        #pdb.set_trace()



parser = argparse.ArgumentParser()
main(parser.parse_args())
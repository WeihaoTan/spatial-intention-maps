
import math
from env_MA import VectorEnv
import argparse
import pdb
import numpy as np
def main(args):


    # Create env
    step = 0
    #env = VectorEnv(use_egl_renderer=True, show_gui=True, room_length=1.0, room_width=0.5, obs_radius = 0.2, termination_step = 2000, target_pos = None, target_width = 0.3)
    env = VectorEnv(use_egl_renderer=True, show_gui=True, room_length=1.0, room_width=0.5, obs_radius = 0.2, termination_step = 2000, target_pos = [0.35, 0.15], target_width = 0.3)


    # Run policy
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
    print("action_spaces", env.action_spaces)
    print("n_action", env.n_action)
    print("get_avail_actions", env.get_avail_actions())
    print("action_space_sample", env.action_space_sample(0), env.action_space_sample(1)) 
    print("macro_action_sample", env.macro_action_sample())

    #pdb.set_trace()

    while True:
        #action = [0.5, 0.25, 0.5, 0.25]
        state, reward, done, _ = env.step(env.macro_action_sample())

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
        print("reward", reward)
        print("--------------------------------------------------------------------")
        
        step += 1
        if done:
            break


parser = argparse.ArgumentParser()
main(parser.parse_args())
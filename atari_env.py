import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from atari_utils import *
MAXIMUM_TIMESTEPS = 100_000


class AtariEnv(Wrapper):
    def __init__(self, 
                 env:gym.Env, 
                 frame_stacking = 4, #how many frames it will be stacked together
                 max_no_op_frames = 30, #maximum number of no op frames at the begining of each episode
                 ):
        super().__init__(env)
        self.frame_stacking = frame_stacking
        self.max_no_op_frames = max_no_op_frames//self.frame_stacking
    
        

    def step(self, action):
        frames = []
        rewards = 0
        truncs = False
        termits = False
        for _ in range(self.frame_stacking):
            frame, reward, terminated, truncated, info = self.env.step(action)
            frame = normilize(frame)
            frames.append(frame)
            rewards+=reward
            if truncated and not truncs:  truncs = True
            if terminated and not termits: termits = True
               

        frames = stack_frames(frames)
        if rewards == 0:
            return frames, rewards, termits, truncs, info
        
        rewards = +1 if rewards > 0 else -1
        return frames, rewards, termits,  truncs, info
    
    
    


    def reset(self, seed = None, **kwargs):
        if seed is not None: obs, info = self.env.reset(seed = seed)
        else : obs, original_info = self.env.reset()
        
        n = np.random.choice(np.arange(self.max_no_op_frames), size = 1)[0]
        n = np.clip(a = n, a_min = self.frame_stacking+1, a_max = self.frame_stacking+self.max_no_op_frames )
        frames = []
        for i in range(n):
            frame, reward, terminated, truncated, info = self.env.step(0)
            if i >= (n-4):
                frames.append(normilize(frame))
        

        if len(frames) != 4:
            print('sth wrong was happend')

        frames = stack_frames(frames)
        return frames, info
    
    
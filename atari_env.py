import gymnasium as gym
from gymnasium import Wrapper
from PIL import Image
import numpy as np
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
        trunc = False
        for _ in range(self.frame_stacking):
            frame, reward, terminated, truncated, info = self.env.step(action)
            frame = self.normilize(frame)
            frames.append(frame)
            rewards+=reward
            if (truncated or terminated) and (not trunc):
                trunc = True

        frames = self.stack_frames(frames)
        return frames, rewards, trunc, info
    
    def normilize(self, frame: np.ndarray):
        if frame.dtype != np.uint8 or frame.shape != (210, 160, 3):
            raise ValueError("Expected RGB frame with shape (210, 160, 3) and dtype uint8.")
        # 1) RGB -> grayscale using standard luma transform
        # (keep as uint8; dot gives float64 so cast back after rounding)
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        gray = np.rint(gray).astype(np.uint8)

        # 2) Resize to 110x84 (PIL expects size=(width, height))
        im = Image.fromarray(gray)  # mode 'L' inferred
        im = im.resize((84, 110), resample=Image.BILINEAR)

        # 3) Crop to 84x84 (center-crop vertically: remove 13 px top & bottom)
        top = (110 - 84) // 2  # 13
        im = im.crop((0, top, 84, top + 84))  # (left, top, right, bottom)

        # Return as uint8 array (H, W)
        return (np.asarray(im, dtype=np.uint8)/255.0).astype(np.float32)
    
    def stack_frames(self, frames, n=4):
        """Simple N-frame stack along channel dim (C, H, W) using your preprocessed single frames."""
        # frames: list of preprocessed (84,84) uint8 arrays
        arr = np.stack(frames[-n:], axis=0)  # (n, 84, 84)
        if arr.shape[0] < n:
            # left-pad with copies of the earliest frame
            pad = np.repeat(arr[:1], n - arr.shape[0], axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        return arr  # (n,84,84)


    def reset(self, seed = None):
        if seed is not None: obs, info = self.env.reset(seed = seed)
        else : obs, original_info = self.env.reset()
        
        n = np.random.choice(np.arange(self.max_no_op_frames), size = 1)
        n = np.clip(a = n, a_min = self.frame_stacking+1, a_max = self.max_no_op_frames )[0]
        frames = []
        for i in range(n):
            frame, reward, terminated, truncated, info = self.env.step(0)
            if i >= (n-4):
                frames.append(self.normilize(frame))
        

        if len(frames) != 4:
            print('sth wrong was happend')

        frames = self.stack_frames(frames)
        return frames, info
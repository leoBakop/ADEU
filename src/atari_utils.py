
import numpy as np
from PIL import Image

def normilize( frame: np.ndarray):
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


def majority_vote_action(q_heads:np.array)->int:
        """
        q_heads: np.array of shape (n_heads, n_actions)
                Each row is Q-values from one head for the current state.
        """
        
        actions = [np.argmax(q) for q in q_heads]
        values, counts = np.unique(actions, return_counts=True)
        action = values[np.argmax(counts)]
        return action

def stack_frames(frames, n=4):
    """Simple N-frame stack along channel dim (C, H, W) using your preprocessed single frames."""
    # frames: list of preprocessed (84,84) uint8 arrays
    arr = np.stack(frames[-n:], axis=0)  # (n, 84, 84)
    if arr.shape[0] < n:
        # left-pad with copies of the earliest frame
        pad = np.repeat(arr[:1], n - arr.shape[0], axis=0)
        arr = np.concatenate([pad, arr], axis=0)
    return arr  # (n,84,84)

import random
from dataclasses import dataclass

import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool

class ReplayBuffer:
    def __init__(self,capacity:int):
        self.capacity=capacity
        self.current_capacity=0
        self.buffer = [None]*capacity
        self.oldest=0
        self.seed=123
        random.seed(self.seed)

    def push(self, transition):
        if self.current_capacity < self.capacity :
            self.buffer[self.current_capacity]=transition
            self.current_capacity+=1
        else:
            self.buffer[self.oldest]=transition
            self.oldest+=1
            if self.oldest == self.capacity :
                self.oldest=0
            

    def sample(self, batch_size):
        if batch_size <=0 :
            raise ValueError("Incompatible Arguments")
        return random.sample(self.buffer[:self.current_capacity],batch_size)

    def __len__(self):
        return self.current_capacity
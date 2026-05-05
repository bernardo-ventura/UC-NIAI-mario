import torch
import torch.nn as nn
import numpy as np
import marioai



class MoveForwardTask(marioai.Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MoveForward"
        
        self.max_x = 0
        self.steps_without_progress = 0

    def reset(self):
        """Reset task state between e/pisodes"""
        super().reset()
        self.max_x = 0
        self.steps_without_progress = 0

    def compute_reward(self, current_obs, last_obs):
        """
        Computes the reward for the current state of the game based on Mario's actions 
        and the environment changes between the current and last observations.
        This function evaluates Mario's progress, interactions with enemies, and overall 
        performance to calculate a reward value. The reward is used as the fitness function for the evolutionary algorithm.
        Parameters:
        - current_obs: The current observation of the game state;
        - last_obs: The previous observation of the game state;
        Returns:
        - reward (float): The computed reward value based on the game state changes.
        Notes for Students:
        - This function is critical for defining the algorithm behavior. The reward function 
          directly impacts the fitness evaluation of the AI.
        - You are encouraged to edit and experiment with this function to design a reward 
          system that aligns with the objectives of the project.
        - Consider the balance between encouraging progress, rewarding kills, and penalizing 
          undesirable behaviors (e.g., cowardice or reckless actions).
        """
        
        # Atualizar progresso máximo
        current_x = current_obs.mario_pos[0]
        if current_x > self.max_x:
          self.max_x = current_x
          self.steps_without_progress = 0
        else:
          self.steps_without_progress += 1
        
        # Fitness baseada principalmente na distância máxima alcançada
        reward = self.max_x/1000
        
        # Penalização fixa se ficou muito tempo parado (sem acumular por frame)
        if self.steps_without_progress > 100:
          reward -= 1  # Penalização fixa por ficar preso
        elif self.steps_without_progress > 50:
          reward -= 0.25  # Penalização moderada por progresso lento
          
        # Bônus por pular (incentiva exploração vertical)
        if last_obs is not None and current_obs.mario_pos[1] < last_obs.mario_pos[1]:
          reward += 2

        # Bonus por movimento no ar (incentiva pulo diagonal)
        if not current_obs.on_ground:
            if last_obs is not None:
                distance = current_obs.mario_pos[0] - last_obs.mario_pos[0]
                if distance > 0:
                    reward += distance  # Movimento no ar vale mais
                  
        # Bônus/penalização por status final
        if current_obs.status == 1:  # Completou o nível
          reward += 1000
        elif current_obs.status == -1:  # Morreu
          reward -= 500
        
        return reward
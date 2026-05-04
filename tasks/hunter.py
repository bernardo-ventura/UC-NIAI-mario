import torch
import torch.nn as nn
import numpy as np
import marioai



class HunterTask(marioai.Task):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.name = "Hunter"

    self.max_x = 0
    self.steps_without_progress = 0
    
  def enemy_still_exists(self, e, curr_enemies):
    if curr_enemies is None:
        return False

    for c in curr_enemies:
        dx = e[0] - c[0]
        dy = e[1] - c[1]
        dist = (dx**2 + dy**2) ** 0.5

        if dist < 1.5:
            return True

    return False

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

    current_x = current_obs.mario_pos[0]
    if current_x > self.max_x:
      self.max_x = current_x
      self.steps_without_progress = 0
    else:
      self.steps_without_progress += 1

    reward = self.max_x/1000

    if self.steps_without_progress > 100:
      reward -= 1  # Penalização fixa por ficar preso
    elif self.steps_without_progress > 50:
      reward -= 0.25 # Penalização moderada por progresso lento

    if last_obs is not None:
      if current_obs.mario_mode < last_obs.mario_mode:
        reward -= 200

    # Kill detection
    if last_obs is not None:
      prev_enemies = last_obs.enemies
      curr_enemies = current_obs.enemies

      if prev_enemies is not None and curr_enemies is not None:
        for e in prev_enemies:
            dist = abs(e[0] - last_obs.mario_pos[0]) # distance to Mario in last frame
            if not self.enemy_still_exists(e, curr_enemies):
              dist = abs(e[0] - last_obs.mario_pos[0])
              
              if dist < 3:
                  reward += 50
              elif dist < 6:
                  reward += 40
              elif dist < 10:
                  reward += 25
              else:
                  reward += 10
      
      scene = current_obs.level_scene
      fireballs = np.sum(scene == 25)

      if fireballs > 0:
        reward += 1

    if last_obs is not None and current_obs.mario_pos[1] < last_obs.mario_pos[1]:
      reward += 2

    # Bonus por movimento no ar (incentiva pulo diagonal)
    if not current_obs.on_ground:
        if last_obs is not None:
            distance = current_obs.mario_pos[0] - last_obs.mario_pos[0]
            if distance > 0:
                reward += distance
    
    # Coin collection bonus
    if last_obs is not None:
      coin_diff = current_obs.coins - last_obs.coins
      if coin_diff > 0:
          reward += coin_diff * 10
    
    # Power-up bonus
    if last_obs is not None:
      if current_obs.mario_mode > last_obs.mario_mode:
          reward += 200

    # Finish level reward
    if current_obs.status == 1:  
      reward += 1000
      
      # Status bonus (fire > big > small)
      if current_obs.mario_mode == 2:  # fire
          reward += 300
      elif current_obs.mario_mode == 1:  # big
          reward += 150

    # Death penalty
    elif current_obs.status == -1:  
      reward -= 500


    return reward
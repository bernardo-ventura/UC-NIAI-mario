
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 1220.5842599999964

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+1] != 15:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

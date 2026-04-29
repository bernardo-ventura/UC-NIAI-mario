
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -211572080.5999992

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_GOOMBA_WINGED:
        if landscape[11+0, 11+-1] == 21:
            action[Mario.KEY_RIGHT] = int(True)
            if can_jump:
                pass
    else:
        action[Mario.KEY_SPEED] = int(True)
        pass
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == 20:
        if enemies[11+-1, 11+1] != Sprite.KIND_GOOMBA_WINGED:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

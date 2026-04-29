
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -334315317.5999989

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+0] == Sprite.KIND_GREEN_KOOPA:
        if landscape[11+0, 11+1] == -10:
            action[Mario.KEY_LEFT] = int(True)
    else:
        pass
        pass
        if landscape[11+0, 11+0] != 0:
            action[Mario.KEY_SPEED] = int(True)
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if enemies[11+1, 11+-1] != Sprite.KIND_GREEN_KOOPA:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+1] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

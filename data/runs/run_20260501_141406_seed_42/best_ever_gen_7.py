
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 291.82462600001713

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if landscape[11+1, 11+1] == 15:
            pass
    else:
        action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+0] != 0:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

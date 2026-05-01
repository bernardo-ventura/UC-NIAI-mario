
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 708.3836899999947

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if landscape[11+1, 11+1] == 25:
            pass
    else:
        action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+-1] != 25:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

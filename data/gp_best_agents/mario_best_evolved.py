
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 267.0

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+0] != Sprite.KIND_SPIKY:
        action[Mario.KEY_RIGHT] = int(True)
        if landscape[11+1, 11+1] != 0:
            action[Mario.KEY_JUMP] = int(True)
        else:
            pass
            pass
        pass
        pass
    else:
        if enemies[11+1, 11+0] != Sprite.KIND_GOOMBA_WINGED:
            pass
        action[Mario.KEY_DOWN] = int(True)

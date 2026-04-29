
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 2763.6

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if landscape[11+0, 11+-1] != -10:
        action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+0, 11+1] != 0:
        action[Mario.KEY_JUMP] = int(True)
        action[Mario.KEY_JUMP] = int(True)
        if enemies[11+1, 11+1] != Sprite.KIND_GREEN_KOOPA_WINGED:
            if landscape[11+1, 11+0] != 14:
                if on_ground:
                    if enemies[11+-1, 11+1] == Sprite.KIND_GOOMBA_WINGED:
                        if landscape[11+1, 11+0] == 15:
                            pass
                            pass
    if enemies[11+0, 11+1] == Sprite.KIND_SPIKY:
        pass
        if enemies[11+0, 11+1] == Sprite.KIND_GREEN_KOOPA:
            action[Mario.KEY_LEFT] = int(True)

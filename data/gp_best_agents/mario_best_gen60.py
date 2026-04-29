
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -454683045.60000074

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+0] == Sprite.KIND_SPIKY_WINGED:
        if landscape[11+0, 11+-1] == -10:
            action[Mario.KEY_JUMP] = int(True)
    else:
        pass
        if enemies[11+0, 11+-1] == Sprite.KIND_GOOMBA_WINGED:
            if enemies[11+0, 11+1] == Sprite.KIND_RED_KOOPA_WINGED:
                if enemies[11+1, 11+1] == Sprite.KIND_FLOWER:
                    if enemies[11+1, 11+1] != Sprite.KIND_SPIKY:
                        if landscape[11+0, 11+-1] != -11:
                            pass
            else:
                action[Mario.KEY_LEFT] = int(True)
        else:
            if enemies[11+1, 11+-1] == Sprite.KIND_SHELL:
                if can_jump:
                    pass
        if enemies[11+-1, 11+-1] == Sprite.KIND_GREEN_KOOPA:
            if on_ground:
                pass
        if on_ground:
            pass
        if landscape[11+0, 11+-1] == -10:
            pass
            action[Mario.KEY_RIGHT] = int(True)
        else:
            if can_jump:
                pass
            else:
                pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if enemies[11+-1, 11+-1] != Sprite.KIND_SHELL:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

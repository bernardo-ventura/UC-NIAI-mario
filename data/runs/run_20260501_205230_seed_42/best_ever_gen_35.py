
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 7447.278440000038

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+0, 11+0] != Sprite.KIND_GREEN_KOOPA:
        if enemies[11+0, 11+-1] != Sprite.KIND_RED_KOOPA:
            if enemies[11+0, 11+0] != Sprite.KIND_GREEN_KOOPA_WINGED:
                action[Mario.KEY_SPEED] = int(True)
        else:
            action[Mario.KEY_JUMP] = int(True)
            if landscape[11+1, 11+-1] == 15:
                action[Mario.KEY_LEFT] = int(True)
            else:
                action[Mario.KEY_RIGHT] = int(True)
        if enemies[11+0, 11+1] == Sprite.KIND_RED_KOOPA:
            action[Mario.KEY_LEFT] = int(True)
        if can_jump:
            if enemies[11+0, 11+-1] != Sprite.KIND_GREEN_KOOPA:
                action[Mario.KEY_JUMP] = int(True)
            pass
        else:
            action[Mario.KEY_RIGHT] = int(True)
            action[Mario.KEY_DOWN] = int(True)
        if on_ground:
            if landscape[11+0, 11+0] == -10:
                if on_ground:
                    pass
        else:
            action[Mario.KEY_JUMP] = int(True)

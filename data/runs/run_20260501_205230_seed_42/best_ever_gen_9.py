
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 2081.033780000016

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+0, 11+0] != Sprite.KIND_RED_KOOPA:
        if enemies[11+0, 11+0] != Sprite.KIND_GREEN_KOOPA:
            pass
        action[Mario.KEY_RIGHT] = int(True)
        if can_jump:
            if enemies[11+-1, 11+-1] != Sprite.KIND_GREEN_KOOPA_WINGED:
                action[Mario.KEY_JUMP] = int(True)
            pass
        else:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_DOWN] = int(True)
        if on_ground:
            pass

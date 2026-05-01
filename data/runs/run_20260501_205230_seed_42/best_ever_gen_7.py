
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 2063.2364600000183

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+0, 11+0] != Sprite.KIND_RED_KOOPA:
        if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
            if enemies[11+0, 11+-1] != Sprite.KIND_BULLET_BILL:
                action[Mario.KEY_SPEED] = int(True)
        else:
            action[Mario.KEY_JUMP] = int(True)
            if landscape[11+1, 11+1] != 0:
                action[Mario.KEY_LEFT] = int(True)
            else:
                if landscape[11+0, 11+-1] == 21:
                    if landscape[11+-1, 11+-1] == 25:
                        pass
        action[Mario.KEY_RIGHT] = int(True)
        if can_jump:
            if enemies[11+-1, 11+-1] != Sprite.KIND_GREEN_KOOPA_WINGED:
                action[Mario.KEY_JUMP] = int(True)
            pass
        else:
            action[Mario.KEY_DOWN] = int(True)
            action[Mario.KEY_DOWN] = int(True)
        if enemies[11+-1, 11+1] == Sprite.KIND_RED_KOOPA:
            if landscape[11+1, 11+1] != -10:
                pass
        else:
            if on_ground:
                action[Mario.KEY_RIGHT] = int(True)

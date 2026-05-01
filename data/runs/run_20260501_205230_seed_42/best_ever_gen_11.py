
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 2089.8464600000193

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
                if landscape[11+0, 11+1] == 21:
                    if landscape[11+-1, 11+-1] == 25:
                        pass
        action[Mario.KEY_SPEED] = int(True)
        if can_jump:
            if enemies[11+-1, 11+-1] != Sprite.KIND_GREEN_KOOPA_WINGED:
                action[Mario.KEY_JUMP] = int(True)
            pass
        else:
            action[Mario.KEY_RIGHT] = int(True)
            action[Mario.KEY_DOWN] = int(True)
        if landscape[11+1, 11+-1] != 20:
            if landscape[11+1, 11+-1] == 25:
                if on_ground:
                    pass
        else:
            action[Mario.KEY_DOWN] = int(True)

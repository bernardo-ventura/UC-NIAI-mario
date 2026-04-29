
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -532815821.10000163

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_RED_KOOPA_WINGED:
        if landscape[11+1, 11+-1] != -10:
            action[Mario.KEY_JUMP] = int(True)
    else:
        pass
        pass
        if enemies[11+0, 11+1] != Sprite.KIND_GOOMBA_WINGED:
            if landscape[11+0, 11+1] != 0:
                if enemies[11+1, 11+-1] == Sprite.KIND_RED_KOOPA_WINGED:
                    pass
                pass
                pass
            else:
                if landscape[11+-1, 11+1] == -10:
                    action[Mario.KEY_LEFT] = int(True)
                else:
                    pass
                    if on_ground:
                        pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+0, 11+1] == -10:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

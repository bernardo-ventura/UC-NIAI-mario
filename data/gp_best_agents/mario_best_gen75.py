
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -569398094.1000019

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+0] == Sprite.KIND_SPIKY_WINGED:
        if landscape[11+0, 11+-1] == -10:
            action[Mario.KEY_JUMP] = int(True)
    else:
        pass
        if enemies[11+0, 11+-1] == Sprite.KIND_GOOMBA_WINGED:
            if enemies[11+0, 11+1] == Sprite.KIND_GOOMBA_WINGED:
                if enemies[11+1, 11+1] == Sprite.KIND_GOOMBA_WINGED:
                    pass
                    action[Mario.KEY_JUMP] = int(True)
                    pass
                    if enemies[11+-1, 11+0] != Sprite.KIND_RED_KOOPA_WINGED:
                        if can_jump:
                            pass
                    else:
                        pass
                    action[Mario.KEY_LEFT] = int(True)
            else:
                action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+1, 11+-1] == Sprite.KIND_SHELL:
                if can_jump:
                    pass
        if enemies[11+-1, 11+1] == Sprite.KIND_GREEN_KOOPA:
            if on_ground:
                pass
        if on_ground:
            pass
        if landscape[11+0, 11+-1] == -10:
            pass
            if landscape[11+1, 11+0] != 0:
                if landscape[11+-1, 11+-1] != -10:
                    if landscape[11+1, 11+0] == 21:
                        pass
                    pass
                    pass
                else:
                    if landscape[11+1, 11+-1] == -11:
                        if enemies[11+0, 11+1] != Sprite.KIND_BULLET_BILL:
                            if landscape[11+-1, 11+-1] != -11:
                                if landscape[11+1, 11+0] == 21:
                                    pass
                                pass
                                pass
                            else:
                                if landscape[11+1, 11+-1] == -11:
                                    if landscape[11+1, 11+0] != -11:
                                        pass
                                else:
                                    pass
                                    if on_ground:
                                        pass
                    else:
                        action[Mario.KEY_RIGHT] = int(True)
            action[Mario.KEY_RIGHT] = int(True)
        else:
            if can_jump:
                pass
            else:
                pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_RIGHT] = int(True)

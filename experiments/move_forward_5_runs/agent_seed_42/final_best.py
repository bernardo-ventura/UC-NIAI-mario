
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 1410.614522173973

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if landscape[11+1, 11+-1] == 0:
        if enemies[11+-1, 11+1] == Sprite.KIND_BULLET_BILL:
            pass
        else:
            if landscape[11+0, 11+-1] == -11:
                if landscape[11+-1, 11+0] != 25:
                    action[Mario.KEY_DOWN] = int(True)
                else:
                    action[Mario.KEY_JUMP] = int(True)
    else:
        if on_ground:
            if landscape[11+1, 11+0] == -11:
                if can_jump:
                    if on_ground:
                        if enemies[11+1, 11+1] == Sprite.KIND_GOOMBA_WINGED:
                            if on_ground:
                                pass
                        else:
                            pass
                    else:
                        if on_ground:
                            pass
                            pass
                        else:
                            action[Mario.KEY_RIGHT] = int(True)
                action[Mario.KEY_DOWN] = int(True)
                if on_ground:
                    pass
                    pass
                else:
                    pass
                pass
            else:
                action[Mario.KEY_LEFT] = int(True)
        else:
            pass
    action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+1] != 15:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+1, 11+1] != Sprite.KIND_BULLET_BILL:
                action[Mario.KEY_DOWN] = int(True)
                if can_jump:
                    pass
                else:
                    action[Mario.KEY_SPEED] = int(True)

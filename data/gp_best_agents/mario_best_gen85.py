
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -641699638.5000024

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_GOOMBA:
        if landscape[11+1, 11+-1] != -10:
            action[Mario.KEY_JUMP] = int(True)
    else:
        pass
        pass
        if enemies[11+0, 11+1] == Sprite.KIND_BULLET_BILL:
            if landscape[11+-1, 11+-1] != 20:
                if landscape[11+1, 11+0] == 21:
                    pass
                pass
                pass
            else:
                pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if landscape[11+1, 11+0] != -11:
            pass
            pass
            if landscape[11+1, 11+0] != 0:
                if landscape[11+-1, 11+-1] != -10:
                    if landscape[11+1, 11+0] == 21:
                        pass
                    pass
                    pass
                else:
                    if landscape[11+1, 11+0] == -11:
                        if landscape[11+0, 11+1] != -11:
                            pass
                    else:
                        pass
                        if on_ground:
                            pass
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_JUMP] = int(True)

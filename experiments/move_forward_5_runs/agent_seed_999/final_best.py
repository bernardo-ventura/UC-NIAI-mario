
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 8062.454488711998

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if can_jump:
        if landscape[11+-1, 11+-1] == 14:
            action[Mario.KEY_SPEED] = int(True)
        else:
            action[Mario.KEY_JUMP] = int(True)
    action[Mario.KEY_SPEED] = int(True)
    if can_jump:
        pass
        if landscape[11+-1, 11+0] != 16:
            if can_jump:
                pass
            else:
                pass
        else:
            if landscape[11+-1, 11+1] == -10:
                if enemies[11+0, 11+1] == Sprite.KIND_FLOWER:
                    pass
                    pass
                    pass
                    if on_ground:
                        pass
                    else:
                        pass
                    if can_jump:
                        pass
                    else:
                        pass
                else:
                    pass
                    pass
        if can_jump:
            pass
        if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA:
            if landscape[11+1, 11+-1] == 16:
                pass
        else:
            if landscape[11+-1, 11+0] != 20:
                pass
        action[Mario.KEY_RIGHT] = int(True)
    else:
        if on_ground:
            action[Mario.KEY_DOWN] = int(True)
        else:
            if landscape[11+0, 11+1] != 20:
                if enemies[11+1, 11+-1] != Sprite.KIND_GREEN_KOOPA:
                    action[Mario.KEY_JUMP] = int(True)
                else:
                    if landscape[11+-1, 11+0] == 20:
                        if on_ground:
                            pass
                    else:
                        if can_jump:
                            pass
                        else:
                            pass
    if can_jump:
        if enemies[11+0, 11+-1] != Sprite.KIND_SPIKY:
            action[Mario.KEY_LEFT] = int(True)
            action[Mario.KEY_SPEED] = int(True)
    else:
        pass
    if enemies[11+0, 11+-1] == Sprite.KIND_FLOWER:
        action[Mario.KEY_SPEED] = int(True)
    else:
        pass
    action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == 15:
        pass
    else:
        action[Mario.KEY_RIGHT] = int(True)
    action[Mario.KEY_RIGHT] = int(True)

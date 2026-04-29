
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -606916484.1000022

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_RED_KOOPA_WINGED:
        if landscape[11+1, 11+-1] != -10:
            action[Mario.KEY_JUMP] = int(True)
    else:
        pass
        pass
        if on_ground:
            if landscape[11+-1, 11+-1] != -11:
                if landscape[11+1, 11+0] == 21:
                    pass
                pass
                pass
            else:
                if landscape[11+1, 11+-1] == -11:
                    if enemies[11+-1, 11+1] != Sprite.KIND_RED_KOOPA_WINGED:
                        if enemies[11+-1, 11+-1] != Sprite.KIND_FLOWER:
                            if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                                if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                                    if landscape[11+-1, 11+1] == -10:
                                        pass
                                        pass
                                else:
                                    if on_ground:
                                        pass
                                    else:
                                        action[Mario.KEY_JUMP] = int(True)
                            else:
                                pass
                else:
                    pass
                    if on_ground:
                        pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if landscape[11+1, 11+0] != -11:
            pass
            pass
            if landscape[11+1, 11+0] != 0:
                if landscape[11+-1, 11+-1] != -10:
                    if landscape[11+1, 11+0] == 21:
                        if landscape[11+1, 11+-1] != 20:
                            pass
                            action[Mario.KEY_JUMP] = int(True)
                            action[Mario.KEY_SPEED] = int(True)
                            pass
                            pass
                            if on_ground:
                                pass
                        else:
                            pass
                            action[Mario.KEY_DOWN] = int(True)
                        if can_jump:
                            pass
                        if landscape[11+-1, 11+1] == -10:
                            if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                                action[Mario.KEY_SPEED] = int(True)
                                action[Mario.KEY_JUMP] = int(True)
                        else:
                            if landscape[11+1, 11+-1] != -11:
                                action[Mario.KEY_RIGHT] = int(True)
                                if landscape[11+0, 11+1] != 0:
                                    action[Mario.KEY_JUMP] = int(True)
                            else:
                                action[Mario.KEY_RIGHT] = int(True)
                        pass
                        pass
                    pass
                    pass
                else:
                    if landscape[11+1, 11+-1] == -11:
                        if landscape[11+0, 11+1] != -10:
                            pass
                    else:
                        pass
                        if on_ground:
                            pass
            pass
            if on_ground:
                pass
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

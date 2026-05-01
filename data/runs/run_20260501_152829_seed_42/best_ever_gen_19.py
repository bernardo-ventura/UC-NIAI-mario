
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 1222.4156999999964

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if landscape[11+0, 11+0] != -11:
            if enemies[11+-1, 11+-1] == Sprite.KIND_RED_KOOPA_WINGED:
                action[Mario.KEY_JUMP] = int(True)
            else:
                pass
                pass
        else:
            if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                pass
                if on_ground:
                    if landscape[11+0, 11+0] == 21:
                        if can_jump:
                            pass
                        else:
                            if can_jump:
                                if on_ground:
                                    pass
                            else:
                                if can_jump:
                                    pass
                                else:
                                    pass
                        pass
                    if on_ground:
                        pass
                    if can_jump:
                        pass
                    else:
                        pass
    else:
        action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+1] != 25:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+-1, 11+1] != Sprite.KIND_SHELL:
                if enemies[11+-1, 11+-1] == Sprite.KIND_SPIKY_WINGED:
                    action[Mario.KEY_DOWN] = int(True)
                else:
                    if landscape[11+1, 11+0] != 25:
                        if can_jump:
                            action[Mario.KEY_JUMP] = int(True)
                        else:
                            action[Mario.KEY_SPEED] = int(True)

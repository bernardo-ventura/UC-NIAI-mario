
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 9311.409676858002

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    action[Mario.KEY_RIGHT] = int(True)
    if enemies[11+-1, 11+-1] != Sprite.KIND_FLOWER:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            pass
    else:
        pass
        pass
    if on_ground:
        if on_ground:
            pass
        else:
            pass
    else:
        pass
        action[Mario.KEY_JUMP] = int(True)
    if landscape[11+-1, 11+1] == -11:
        pass
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if enemies[11+-1, 11+0] == Sprite.KIND_SPIKY:
            if landscape[11+0, 11+0] == 25:
                if can_jump:
                    pass
                    pass
                else:
                    pass
                if can_jump:
                    pass
                else:
                    pass
                pass
            else:
                if enemies[11+1, 11+-1] != Sprite.KIND_RED_KOOPA:
                    pass
    else:
        if enemies[11+0, 11+-1] == Sprite.KIND_GREEN_KOOPA:
            if enemies[11+0, 11+-1] == Sprite.KIND_FLOWER:
                pass
        else:
            if landscape[11+-1, 11+-1] == 0:
                if landscape[11+0, 11+-1] != 25:
                    if can_jump:
                        if can_jump:
                            if on_ground:
                                if landscape[11+-1, 11+1] == -10:
                                    action[Mario.KEY_SPEED] = int(True)
                            else:
                                pass
                    else:
                        if landscape[11+-1, 11+-1] == -11:
                            action[Mario.KEY_RIGHT] = int(True)
                        else:
                            if can_jump:
                                pass
                            else:
                                if landscape[11+1, 11+0] != -11:
                                    if landscape[11+1, 11+1] == 21:
                                        action[Mario.KEY_LEFT] = int(True)
                                else:
                                    if enemies[11+-1, 11+0] != Sprite.KIND_GOOMBA_WINGED:
                                        if enemies[11+1, 11+0] == Sprite.KIND_GOOMBA:
                                            pass
                                    else:
                                        if on_ground:
                                            pass
                                        else:
                                            pass
                    if on_ground:
                        if on_ground:
                            if on_ground:
                                pass
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                else:
                    action[Mario.KEY_JUMP] = int(True)
            else:
                action[Mario.KEY_LEFT] = int(True)
                if enemies[11+1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                    pass
                if enemies[11+0, 11+-1] == Sprite.KIND_GOOMBA:
                    pass
                    pass
        if can_jump:
            if can_jump:
                pass
        else:
            pass
        if enemies[11+0, 11+0] != Sprite.KIND_SHELL:
            pass
        else:
            action[Mario.KEY_SPEED] = int(True)

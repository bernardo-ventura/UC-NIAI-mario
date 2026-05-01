
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 2098.03675000002

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+0] != Sprite.KIND_SPIKY:
        if enemies[11+0, 11+0] != Sprite.KIND_RED_KOOPA_WINGED:
            if enemies[11+-1, 11+0] == Sprite.KIND_GREEN_KOOPA:
                pass
                if enemies[11+0, 11+-1] == Sprite.KIND_FLOWER:
                    action[Mario.KEY_SPEED] = int(True)
                else:
                    pass
        else:
            if landscape[11+-1, 11+0] != 16:
                action[Mario.KEY_DOWN] = int(True)
                if on_ground:
                    pass
            else:
                action[Mario.KEY_DOWN] = int(True)
        action[Mario.KEY_SPEED] = int(True)
        if can_jump:
            if enemies[11+0, 11+-1] != Sprite.KIND_RED_KOOPA:
                action[Mario.KEY_JUMP] = int(True)
            pass
        else:
            action[Mario.KEY_RIGHT] = int(True)
            action[Mario.KEY_DOWN] = int(True)
        if landscape[11+1, 11+-1] != 20:
            if on_ground:
                action[Mario.KEY_RIGHT] = int(True)
                if landscape[11+1, 11+1] == -11:
                    if can_jump:
                        if enemies[11+-1, 11+0] != Sprite.KIND_GREEN_KOOPA_WINGED:
                            if on_ground:
                                pass
                        else:
                            if can_jump:
                                pass
                    pass
                else:
                    if on_ground:
                        pass
                    else:
                        pass
                        pass
                        if on_ground:
                            pass
                        else:
                            pass
                        pass
        else:
            action[Mario.KEY_DOWN] = int(True)

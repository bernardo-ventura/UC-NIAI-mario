
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 726.673325582001

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if landscape[11+1, 11+-1] == 25:
        if landscape[11+-1, 11+0] == 15:
            if can_jump:
                if can_jump:
                    pass
                else:
                    pass
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
        else:
            pass
            if enemies[11+-1, 11+1] != Sprite.KIND_GREEN_KOOPA:
                pass
                pass
                pass
                pass
            if enemies[11+1, 11+1] != Sprite.KIND_SHELL:
                if on_ground:
                    pass
                else:
                    pass
                pass
    else:
        if can_jump:
            if on_ground:
                pass
                if enemies[11+-1, 11+0] != Sprite.KIND_GREEN_KOOPA_WINGED:
                    if landscape[11+1, 11+1] == 16:
                        action[Mario.KEY_RIGHT] = int(True)
                else:
                    action[Mario.KEY_DOWN] = int(True)
                action[Mario.KEY_RIGHT] = int(True)
                action[Mario.KEY_JUMP] = int(True)
            else:
                pass
        else:
            if can_jump:
                if can_jump:
                    action[Mario.KEY_JUMP] = int(True)
                else:
                    pass
            else:
                if landscape[11+1, 11+-1] == 0:
                    if enemies[11+0, 11+-1] == Sprite.KIND_RED_KOOPA:
                        pass
                    else:
                        pass
                action[Mario.KEY_SPEED] = int(True)
                if landscape[11+1, 11+-1] == 0:
                    if enemies[11+1, 11+0] != Sprite.KIND_RED_KOOPA_WINGED:
                        if can_jump:
                            pass
                    else:
                        if on_ground:
                            pass
                        else:
                            action[Mario.KEY_RIGHT] = int(True)

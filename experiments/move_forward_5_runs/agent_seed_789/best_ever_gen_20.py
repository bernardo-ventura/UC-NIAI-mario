
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 4756.803174848001

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if landscape[11+0, 11+-1] == 25:
        if landscape[11+-1, 11+0] == 0:
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
                if enemies[11+-1, 11+-1] != Sprite.KIND_GOOMBA:
                    if on_ground:
                        if can_jump:
                            if landscape[11+1, 11+1] != 14:
                                pass
                            else:
                                pass
                                pass
                    else:
                        action[Mario.KEY_JUMP] = int(True)
                    if landscape[11+0, 11+1] != 0:
                        pass
            if enemies[11+1, 11+1] != Sprite.KIND_SHELL:
                if on_ground:
                    pass
                else:
                    pass
                action[Mario.KEY_SPEED] = int(True)
    else:
        if can_jump:
            if enemies[11+-1, 11+0] != Sprite.KIND_GREEN_KOOPA_WINGED:
                if landscape[11+1, 11+0] == 16:
                    action[Mario.KEY_RIGHT] = int(True)
            else:
                if enemies[11+0, 11+-1] != Sprite.KIND_SPIKY_WINGED:
                    action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_RIGHT] = int(True)
            action[Mario.KEY_JUMP] = int(True)
        else:
            if can_jump:
                if can_jump:
                    action[Mario.KEY_DOWN] = int(True)
                else:
                    pass
            else:
                if on_ground:
                    if enemies[11+1, 11+1] != Sprite.KIND_FLOWER:
                        if landscape[11+1, 11+1] != 14:
                            pass
                        else:
                            pass
                            pass
                else:
                    action[Mario.KEY_JUMP] = int(True)
                action[Mario.KEY_SPEED] = int(True)
                action[Mario.KEY_RIGHT] = int(True)


# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 9807.44781500005

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if landscape[11+0, 11+0] != -11:
            if enemies[11+-1, 11+-1] != Sprite.KIND_GREEN_KOOPA:
                action[Mario.KEY_SPEED] = int(True)
            else:
                if landscape[11+1, 11+1] != 15:
                    if landscape[11+1, 11+0] == 20:
                        pass
                        if landscape[11+1, 11+0] == 20:
                            action[Mario.KEY_LEFT] = int(True)
                    else:
                        action[Mario.KEY_LEFT] = int(True)
                else:
                    action[Mario.KEY_DOWN] = int(True)
                pass
        else:
            if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                if landscape[11+0, 11+1] == 16:
                    action[Mario.KEY_JUMP] = int(True)
    else:
        if enemies[11+-1, 11+-1] != Sprite.KIND_FLOWER:
            if enemies[11+1, 11+1] == Sprite.KIND_GOOMBA:
                action[Mario.KEY_LEFT] = int(True)
            if enemies[11+1, 11+1] == Sprite.KIND_SPIKY_WINGED:
                if landscape[11+1, 11+1] == -11:
                    if landscape[11+1, 11+0] != 0:
                        pass
                    else:
                        pass
                else:
                    if on_ground:
                        pass
                    else:
                        pass
                        pass
            else:
                if on_ground:
                    pass
                else:
                    if enemies[11+1, 11+-1] == Sprite.KIND_BULLET_BILL:
                        pass
                        pass
                    if can_jump:
                        if enemies[11+1, 11+0] != Sprite.KIND_GREEN_KOOPA_WINGED:
                            action[Mario.KEY_SPEED] = int(True)
                        pass
                    if enemies[11+0, 11+1] != Sprite.KIND_GOOMBA:
                        if on_ground:
                            pass
                            if enemies[11+-1, 11+-1] == Sprite.KIND_GREEN_KOOPA_WINGED:
                                if enemies[11+1, 11+1] != Sprite.KIND_RED_KOOPA:
                                    if can_jump:
                                        action[Mario.KEY_SPEED] = int(True)
                                    if enemies[11+-1, 11+-1] == Sprite.KIND_SHELL:
                                        pass
                                    if on_ground:
                                        pass
                                    else:
                                        if landscape[11+0, 11+0] == 25:
                                            if can_jump:
                                                pass
                                            else:
                                                if can_jump:
                                                    pass
                                                else:
                                                    pass
                                    if enemies[11+0, 11+1] != Sprite.KIND_RED_KOOPA_WINGED:
                                        pass
                                        pass
                            else:
                                pass
                        else:
                            if landscape[11+0, 11+1] != -11:
                                action[Mario.KEY_JUMP] = int(True)
                            pass
                    else:
                        action[Mario.KEY_LEFT] = int(True)
                    action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+1] != 25:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+-1, 11+-1] != Sprite.KIND_SHELL:
                if landscape[11+1, 11+-1] == 25:
                    if enemies[11+-1, 11+1] == Sprite.KIND_SHELL:
                        if landscape[11+0, 11+-1] != 0:
                            action[Mario.KEY_RIGHT] = int(True)
                else:
                    if landscape[11+1, 11+1] != 20:
                        if enemies[11+0, 11+-1] != Sprite.KIND_SHELL:
                            if landscape[11+-1, 11+0] != -11:
                                if landscape[11+-1, 11+0] == 0:
                                    pass
                                    if on_ground:
                                        pass


# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 8152.7966600000345

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if landscape[11+0, 11+0] != -11:
            if enemies[11+-1, 11+0] == Sprite.KIND_SHELL:
                pass
                action[Mario.KEY_JUMP] = int(True)
                pass
                pass
        else:
            if enemies[11+-1, 11+-1] != Sprite.KIND_RED_KOOPA_WINGED:
                pass
                if on_ground:
                    if can_jump:
                        pass
                    else:
                        if can_jump:
                            if on_ground:
                                pass
                        else:
                            if enemies[11+-1, 11+0] == Sprite.KIND_SPIKY_WINGED:
                                pass
                            else:
                                pass
    else:
        if enemies[11+-1, 11+-1] != Sprite.KIND_FLOWER:
            if enemies[11+1, 11+-1] == Sprite.KIND_FLOWER:
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
                        action[Mario.KEY_RIGHT] = int(True)
                    if can_jump:
                        if landscape[11+0, 11+0] == 25:
                            action[Mario.KEY_SPEED] = int(True)
                        pass
                    if enemies[11+0, 11+1] != Sprite.KIND_GOOMBA:
                        if on_ground:
                            pass
                            if enemies[11+-1, 11+-1] != Sprite.KIND_GREEN_KOOPA_WINGED:
                                if on_ground:
                                    pass
                            else:
                                pass
                        else:
                            if landscape[11+0, 11+1] != 15:
                                action[Mario.KEY_JUMP] = int(True)
                            pass
                    else:
                        action[Mario.KEY_DOWN] = int(True)
                    action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+1] != 25:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+1, 11+0] != Sprite.KIND_RED_KOOPA:
                if enemies[11+-1, 11+-1] == Sprite.KIND_SPIKY_WINGED:
                    action[Mario.KEY_DOWN] = int(True)
                else:
                    if landscape[11+1, 11+1] != 25:
                        if can_jump:
                            action[Mario.KEY_SPEED] = int(True)
                        else:
                            action[Mario.KEY_SPEED] = int(True)

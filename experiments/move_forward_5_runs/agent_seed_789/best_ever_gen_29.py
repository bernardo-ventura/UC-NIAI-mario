
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 8018.116816161995

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if landscape[11+1, 11+-1] == 25:
        if landscape[11+-1, 11+0] == 0:
            if can_jump:
                if can_jump:
                    if enemies[11+0, 11+1] == Sprite.KIND_GOOMBA_WINGED:
                        if enemies[11+0, 11+1] == Sprite.KIND_SPIKY:
                            if enemies[11+-1, 11+1] != Sprite.KIND_RED_KOOPA_WINGED:
                                if can_jump:
                                    pass
                                else:
                                    action[Mario.KEY_RIGHT] = int(True)
                            else:
                                pass
                else:
                    if enemies[11+-1, 11+0] == Sprite.KIND_BULLET_BILL:
                        if enemies[11+-1, 11+1] == Sprite.KIND_SHELL:
                            if can_jump:
                                pass
                        else:
                            if landscape[11+-1, 11+-1] == 0:
                                if enemies[11+0, 11+-1] == Sprite.KIND_SPIKY:
                                    action[Mario.KEY_DOWN] = int(True)
                                else:
                                    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
                                        pass
                                    else:
                                        pass
                            else:
                                pass
            if enemies[11+0, 11+0] == Sprite.KIND_RED_KOOPA:
                if landscape[11+1, 11+0] != 14:
                    action[Mario.KEY_DOWN] = int(True)
                    pass
            action[Mario.KEY_DOWN] = int(True)
            if on_ground:
                pass
                pass
            else:
                if enemies[11+1, 11+-1] == Sprite.KIND_BULLET_BILL:
                    if on_ground:
                        pass
                    else:
                        pass
                else:
                    if can_jump:
                        pass
                if can_jump:
                    pass
                else:
                    pass
                pass
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
                        action[Mario.KEY_RIGHT] = int(True)
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
            if enemies[11+-1, 11+0] != Sprite.KIND_GREEN_KOOPA:
                if landscape[11+0, 11+0] == 16:
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
                        action[Mario.KEY_DOWN] = int(True)
                else:
                    action[Mario.KEY_JUMP] = int(True)
                action[Mario.KEY_SPEED] = int(True)
                action[Mario.KEY_RIGHT] = int(True)

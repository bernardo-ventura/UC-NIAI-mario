
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -750925713.5000029

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_GOOMBA:
        if landscape[11+0, 11+-1] != 0:
            action[Mario.KEY_JUMP] = int(True)
    else:
        pass
        pass
        if enemies[11+0, 11+-1] != Sprite.KIND_BULLET_BILL:
            if landscape[11+1, 11+-1] != -11:
                pass
                pass
            else:
                if landscape[11+1, 11+-1] == -11:
                    if landscape[11+-1, 11+0] != -11:
                        pass
                else:
                    pass
                    if on_ground:
                        pass
        action[Mario.KEY_SPEED] = int(True)
        if enemies[11+0, 11+-1] == Sprite.KIND_GOOMBA_WINGED:
            if enemies[11+0, 11+1] == Sprite.KIND_SHELL:
                if enemies[11+0, 11+-1] == Sprite.KIND_RED_KOOPA_WINGED:
                    if enemies[11+1, 11+1] == Sprite.KIND_RED_KOOPA_WINGED:
                        if landscape[11+0, 11+-1] == -11:
                            pass
            else:
                action[Mario.KEY_JUMP] = int(True)
        else:
            if landscape[11+1, 11+-1] == 0:
                if can_jump:
                    pass
        if enemies[11+-1, 11+1] == Sprite.KIND_GOOMBA_WINGED:
            if on_ground:
                pass
        if landscape[11+1, 11+0] != -10:
            pass
        if landscape[11+0, 11+-1] == -10:
            pass
            pass
        else:
            if can_jump:
                pass
            else:
                pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if enemies[11+-1, 11+-1] != Sprite.KIND_BULLET_BILL:
            action[Mario.KEY_JUMP] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+1] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_JUMP] = int(True)

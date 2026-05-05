
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 8036.908622031997

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    action[Mario.KEY_RIGHT] = int(True)
    if enemies[11+-1, 11+0] != Sprite.KIND_FLOWER:
        pass
    else:
        pass
    if landscape[11+-1, 11+0] == 16:
        action[Mario.KEY_SPEED] = int(True)
    else:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
            if on_ground:
                if enemies[11+0, 11+0] == Sprite.KIND_GOOMBA_WINGED:
                    pass
            else:
                action[Mario.KEY_SPEED] = int(True)
                action[Mario.KEY_JUMP] = int(True)
        if can_jump:
            pass
        if landscape[11+-1, 11+0] == 21:
            pass
        if on_ground:
            if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
                if enemies[11+-1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
                    action[Mario.KEY_LEFT] = int(True)
                else:
                    action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+-1, 11+-1] != Sprite.KIND_GREEN_KOOPA:
                pass
            if enemies[11+1, 11+-1] == Sprite.KIND_GOOMBA:
                pass
            else:
                pass
            if enemies[11+-1, 11+-1] != Sprite.KIND_SPIKY_WINGED:
                if can_jump:
                    action[Mario.KEY_SPEED] = int(True)
                else:
                    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
                        pass
                    else:
                        action[Mario.KEY_SPEED] = int(True)
                pass
        if on_ground:
            if landscape[11+0, 11+0] != 25:
                action[Mario.KEY_LEFT] = int(True)
                if enemies[11+-1, 11+1] == Sprite.KIND_GOOMBA_WINGED:
                    if landscape[11+-1, 11+-1] != -10:
                        pass
                    else:
                        pass
                else:
                    pass
                    pass
        else:
            if landscape[11+1, 11+-1] != 15:
                if can_jump:
                    pass
                action[Mario.KEY_JUMP] = int(True)
            else:
                action[Mario.KEY_SPEED] = int(True)
                action[Mario.KEY_SPEED] = int(True)
            if enemies[11+0, 11+0] == Sprite.KIND_BULLET_BILL:
                pass
            if on_ground:
                pass
            else:
                pass
            if enemies[11+-1, 11+1] == Sprite.KIND_RED_KOOPA_WINGED:
                action[Mario.KEY_RIGHT] = int(True)
            else:
                pass
                action[Mario.KEY_RIGHT] = int(True)
                if landscape[11+1, 11+-1] != 20:
                    pass
        pass

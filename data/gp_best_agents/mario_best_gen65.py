
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -495396939.6000012

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_SHELL:
        if landscape[11+1, 11+-1] != -11:
            action[Mario.KEY_LEFT] = int(True)
    else:
        pass
        pass
        if enemies[11+0, 11+-1] == Sprite.KIND_BULLET_BILL:
            if landscape[11+-1, 11+-1] != -10:
                if landscape[11+1, 11+0] == 21:
                    pass
                pass
                pass
            else:
                if enemies[11+-1, 11+1] == Sprite.KIND_GREEN_KOOPA:
                    if on_ground:
                        pass
                else:
                    pass
                    if on_ground:
                        pass
        action[Mario.KEY_SPEED] = int(True)
    if landscape[11+-1, 11+1] == -10:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_RIGHT] = int(True)


# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -157427315.09999946

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+-1, 11+1] == Sprite.KIND_GREEN_KOOPA:
        if landscape[11+0, 11+-1] == -10:
            action[Mario.KEY_LEFT] = int(True)
    else:
        action[Mario.KEY_SPEED] = int(True)
        pass
        pass
    if landscape[11+-1, 11+1] == 20:
        if enemies[11+-1, 11+1] != Sprite.KIND_SHELL:
            action[Mario.KEY_SPEED] = int(True)
            action[Mario.KEY_JUMP] = int(True)
    else:
        if landscape[11+1, 11+0] != -11:
            action[Mario.KEY_RIGHT] = int(True)
            if landscape[11+0, 11+1] != 0:
                action[Mario.KEY_JUMP] = int(True)
        else:
            action[Mario.KEY_SPEED] = int(True)

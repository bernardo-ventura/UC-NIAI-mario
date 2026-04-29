
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 2763.6

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if landscape[11+0, 11+-1] != -10:
        action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+0, 11+1] != 0:
        if landscape[11+-1, 11+0] == 0:
            if enemies[11+0, 11+1] != Sprite.KIND_GOOMBA_WINGED:
                if landscape[11+1, 11+0] != 14:
                    pass
                else:
                    if landscape[11+0, 11+-1] != 21:
                        pass
            if landscape[11+0, 11+1] != 25:
                action[Mario.KEY_JUMP] = int(True)
                if on_ground:
                    pass
                    pass
                else:
                    pass
                    pass
                if can_jump:
                    pass
        else:
            action[Mario.KEY_RIGHT] = int(True)
        if enemies[11+1, 11+0] != Sprite.KIND_FLOWER:
            if enemies[11+1, 11+-1] == Sprite.KIND_SHELL:
                pass
                if on_ground:
                    pass
                else:
                    pass
                pass
            else:
                pass
    if landscape[11+1, 11+-1] == 15:
        pass
        if landscape[11+0, 11+-1] != -10:
            action[Mario.KEY_LEFT] = int(True)

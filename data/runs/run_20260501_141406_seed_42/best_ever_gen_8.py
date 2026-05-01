
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 677.4294899999987

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    action[Mario.KEY_SPEED] = int(True)
    action[Mario.KEY_RIGHT] = int(True)
    if landscape[11+1, 11+0] != 15:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        else:
            if enemies[11+-1, 11+1] == Sprite.KIND_SPIKY_WINGED:
                pass
                if can_jump:
                    pass
                else:
                    pass

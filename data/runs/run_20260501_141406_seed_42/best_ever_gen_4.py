
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 230.7452700000162

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+1] == Sprite.KIND_GREEN_KOOPA_WINGED:
        if landscape[11+1, 11+1] == 0:
            pass
    else:
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

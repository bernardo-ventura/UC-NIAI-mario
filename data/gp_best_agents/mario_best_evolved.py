
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -119697.00000000071

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+0, 11+-1] == Sprite.KIND_BULLET_BILL:
        if landscape[11+0, 11+1] != 0:
            action[Mario.KEY_SPEED] = int(True)
    else:
        action[Mario.KEY_RIGHT] = int(True)

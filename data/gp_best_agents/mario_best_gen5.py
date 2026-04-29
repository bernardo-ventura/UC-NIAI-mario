
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: -80297089.49999993

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+0, 11+-1] == Sprite.KIND_SPIKY:
        if enemies[11+1, 11+0] != Sprite.KIND_SHELL:
            pass
        else:
            pass
    if enemies[11+1, 11+0] != Sprite.KIND_GOOMBA:
        if enemies[11+1, 11+0] == Sprite.KIND_SPIKY:
            if enemies[11+-1, 11+1] == Sprite.KIND_SPIKY:
                action[Mario.KEY_SPEED] = int(True)

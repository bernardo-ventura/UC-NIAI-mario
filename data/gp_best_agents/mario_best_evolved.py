
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: 267.0

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if enemies[11+1, 11+0] != Sprite.KIND_RED_KOOPA:
        if enemies[11+0, 11+1] != Sprite.KIND_SPIKY_WINGED:
            action[Mario.KEY_RIGHT] = int(True)

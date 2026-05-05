def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
    if can_jump:
        if can_jump:
            action[Mario.KEY_JUMP] = int(True)
        action[Mario.KEY_RIGHT] = int(True)

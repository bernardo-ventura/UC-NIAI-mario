"""
Mario AI - Evolutionary Genetic Programming
============================================
Implementação de um Algoritmo Evolutivo usando Programação Genética
para evoluir controladores autônomos para Super Mario.
"""

import operator
import random
import numpy as np
import sys
import textwrap
import pickle
import copy
import json
from pathlib import Path

import time
from datetime import datetime

# USER IMPORTS (Assuming evaluate is provided in your evaluation.py)
from evaluation import evaluate

# -----------------------------------------------------------------------------
# USER IMPORTS / MOCKS
# -----------------------------------------------------------------------------
try:
    import marioai
    from agents import CodeAgent, Mario, Sprite
except ImportError:
    # Mocks for standalone testing if libraries are missing
    class Mario:
        KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_JUMP, KEY_SPEED = 0, 1, 2, 3, 4
    class Sprite:
        KIND_GOOMBA = 80
        KIND_GOOMBA_WINGED = 81
        KIND_RED_KOOPA = 82
        KIND_RED_KOOPA_WINGED = 83
        KIND_GREEN_KOOPA = 84
        KIND_GREEN_KOOPA_WINGED = 85
        KIND_BULLET_BILL = 86
        KIND_SPIKY = 87
        KIND_SPIKY_WINGED = 88
    class CodeAgent: pass
    print("Warning: marioai/agents modules not found. Using mocks.")

from deap import base, creator, tools, gp

# -----------------------------------------------------------------------------
# 0. HELPER: Safe Generator
# -----------------------------------------------------------------------------
def safe_gen_grow(pset, min_, max_, type_=None):
    """
    Generates a random GP tree respecting type constraints.
    Uses the 'Grow' method with min/max depth limits.
    """
    if type_ is None: type_ = pset.ret
    expr = []
    stack = [(0, type_)]
    while stack:
        depth, type_ = stack.pop()
        try: has_primitives = len(pset.primitives[type_]) > 0
        except KeyError: has_primitives = False
        try: has_terminals = len(pset.terminals[type_]) > 0
        except KeyError: has_terminals = False
        
        if not has_terminals and not has_primitives:
            raise IndexError(f"Type '{type_.__name__}' has no primitives/terminals!")

        should_grow = False
        if not has_terminals: should_grow = True
        elif not has_primitives: should_grow = False
        else:
            if depth < min_: should_grow = True
            elif depth >= max_: should_grow = False
            else: should_grow = (random.random() < 0.5)

        if should_grow:
            prim = random.choice(pset.primitives[type_])
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
        else:
            term = random.choice(pset.terminals[type_])
            if isinstance(term, type): term = term()
            expr.append(term)
    return expr

def indent(text):
    """Adds 4-space indentation to each line of text."""
    return "\n".join("    " + line for line in text.split("\n"))

# -----------------------------------------------------------------------------
# 1. TYPE DEFINITIONS
# -----------------------------------------------------------------------------
class Expr: pass        # Expression/statement
class Condition: pass   # Boolean condition for if-statements
class Key: pass         # Controller key (LEFT, RIGHT, etc.)
class Bool: pass        # Boolean value (True/False)
class Position: pass      # (x, y) coordinates
class Comparator: pass    # ==, !==
class EnemyType: pass    # Type of enemy (Goomba, Koopa, etc.)
class LandscapeType: pass # Type of landscape (Ground, Air, etc.)

# -----------------------------------------------------------------------------
# 2. PRIMITIVES: STRING BUILDERS
# -----------------------------------------------------------------------------
def str_if_then(cond, expr):
    """Builds an if-then statement."""
    return f"if {cond}:\n{indent(expr)}"

def str_sequence(expr1, expr2):
    """Sequences two expressions."""
    return f"{expr1}\n{expr2}"

def str_set_action(key, val):
    """Sets an action key to a value."""
    return f"action[{key}] = int({val})"

def str_check_enemy(pos_x, pos_y, comp, enemy_type):
    """Checks for a specific enemy type at a given position."""
    return f"enemies[11+{pos_x}, 11+{pos_y}] {comp} {enemy_type}"

def str_if_then_else(cond, expr_true, expr_false):
    """Builds an if-then-else statement."""
    return f"if {cond}:\n{indent(expr_true)}\nelse:\n{indent(expr_false)}"

def str_check_landscape(pos_x, pos_y, comp, landscape_type):
    """Checks for a specific landscape type at a given position."""
    return f"landscape[11+{pos_x}, 11+{pos_y}] {comp} {landscape_type}"

# -----------------------------------------------------------------------------
# 3. GRAMMAR CONFIGURATION
# -----------------------------------------------------------------------------
pset = gp.PrimitiveSetTyped("MAIN", [], Expr)

# Core Logic Primitives
pset.addPrimitive(str_if_then, [Condition, Expr], Expr)
pset.addPrimitive(str_if_then_else, [Condition, Expr, Expr], Expr)
pset.addPrimitive(str_sequence, [Expr, Expr], Expr)
pset.addPrimitive(str_set_action, [Key, Bool], Expr)
pset.addPrimitive(str_check_enemy, [Position, Position, Comparator, EnemyType], Condition)
pset.addPrimitive(str_check_landscape, [Position, Position, Comparator, LandscapeType], Condition)
pset.addTerminal("pass", Expr, name="NoOp")

# Basic Senses (Provided directly by the environment variables)
pset.addTerminal("on_ground", Condition, name="IsMarioOnGround")
pset.addTerminal("can_jump", Condition, name="MayMarioJump")

# Constants
pset.addTerminal("True", Bool)

# Limited Actions (Only Right and Jump for now - TO BE EXPANDED)
pset.addTerminal("Mario.KEY_RIGHT", Key, name="RIGHT")
pset.addTerminal("Mario.KEY_JUMP", Key, name="JUMP")
pset.addTerminal("Mario.KEY_LEFT", Key, name="LEFT")
pset.addTerminal("Mario.KEY_DOWN", Key, name="DOWN")
pset.addTerminal("Mario.KEY_SPEED", Key, name="SPEED")

# Terminal for comparators
pset.addTerminal("==", Comparator, name="Equals")
pset.addTerminal("!=", Comparator, name="NotEquals")

# Terminals for positions
pset.addTerminal(-1, Position, name="PosNeg")    # Esquerda/Acima
pset.addTerminal(0, Position, name="PosZero")    # Centro
pset.addTerminal(1, Position, name="PosPos")     # Direita/Abaixo

# Terminals for enemy types
pset.addTerminal("Sprite.KIND_GOOMBA", EnemyType, name="Goomba")
pset.addTerminal("Sprite.KIND_GOOMBA_WINGED", EnemyType, name="GoombaWinged")
pset.addTerminal("Sprite.KIND_RED_KOOPA", EnemyType, name="RedKoopa")
pset.addTerminal("Sprite.KIND_RED_KOOPA_WINGED", EnemyType, name="RedKoopaWinged")
pset.addTerminal("Sprite.KIND_GREEN_KOOPA", EnemyType, name="GreenKoopa")
pset.addTerminal("Sprite.KIND_GREEN_KOOPA_WINGED", EnemyType, name="GreenKoopaWinged")
pset.addTerminal("Sprite.KIND_BULLET_BILL", EnemyType, name="BulletBill")
pset.addTerminal("Sprite.KIND_SPIKY", EnemyType, name="Spiky")
pset.addTerminal("Sprite.KIND_SPIKY_WINGED", EnemyType, name="SpikyWinged")
pset.addTerminal("Sprite.KIND_FLOWER", EnemyType, name="Flower")
pset.addTerminal("Sprite.KIND_SHELL", EnemyType, name="Shell")

# Terminals for landscape types
pset.addTerminal("0", LandscapeType, name="Empty")
pset.addTerminal("-11", LandscapeType, name="SoftObstacle")
pset.addTerminal("-10", LandscapeType, name="HardObstacle")
pset.addTerminal("14", LandscapeType, name="Mushroom")
pset.addTerminal("15", LandscapeType, name="FireFlower")   
pset.addTerminal("25", LandscapeType, name="Fireball") 
pset.addTerminal("16", LandscapeType, name="Brick")
pset.addTerminal("21", LandscapeType, name="QuestionBrick")
pset.addTerminal("20", LandscapeType, name="EnemyObstacle")


# TODO: Add enemy detection primitives
# TODO: Add landscape detection primitives
# TODO: Add if-then-else primitive

# -----------------------------------------------------------------------------
# 4. EVOLUTIONARY ALGORITHM SETUP
# -----------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", safe_gen_grow, pset=pset, min_=3, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)

# DONE: Register evolutionary operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

def evaluate_gp_individual(individual):
    """
    Converts a GP tree individual into executable Python code and evaluates it.
    
    Args:
        individual: A DEAP PrimitiveTree representing a Mario controller
        
    Returns:
        fitness_score: Float representing the agent's performance
    """
    code_body = toolbox.compile(individual)
    agent_prototype = CodeAgent
    full_code_str = f"""
def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
{indent(code_body)}
""" 
    try:
        reward = evaluate(agent_prototype, full_code_str)
    except NameError:
        # If your evaluation isn't loaded properly, mock a random score for testing
        print(" [Sim] Evaluation not linked properly. Returning 0 score.")
        reward = 0
        
    return reward



# -----------------------------------------------------------------------------
# 5. PERSISTENCE HELPERS
# -----------------------------------------------------------------------------
def save_best_individual(best_ind, toolbox, filename_py="mario_best_evolved.py"):
    """
    Saves the best individual as a readable Python script.
    
    Args:
        best_ind: Best individual from the population
        toolbox: DEAP toolbox for compilation
        filename_py: Output filename
    """
    if best_ind is None:
        print("No individual to save.")
        return

    code_body = toolbox.compile(best_ind)
    fitness_val = best_ind.fitness.values[0] if best_ind.fitness.valid else "Unknown"
    
    full_code = f"""
# Evolved Mario Controller (Evolutionary Algorithm)
# Fitness: {fitness_val}

def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):
{indent(code_body)}
"""
    output_path = Path(filename_py)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(full_code)
    print(f"Saved executable code to '{output_path}'")
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION: EVOLUTIONARY ALGORITHM
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python mario_evolutionary_gp.py <seed>")
        sys.exit(1)

    seed = int(sys.argv[1])
    random.seed(seed)

    # ===== RUN SETUP =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data/runs") / f"run_{timestamp}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "experiment.log"
    config_file = run_dir / "config.json"

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    # ===== PARAMETERS =====
    GENERATIONS = 30
    POP_SIZE = 100
    CXPB = 0.8
    MUTPB = 0.8

    config_data = {
        "seed": seed,
        "generations": GENERATIONS,
        "population_size": POP_SIZE,
        "crossover_prob": CXPB,
        "mutation_prob": MUTPB
    }

    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    # ===== EXPERIMENT DATA =====
    experiment_data = {
        "seed": seed,
        "generations": [],
        "final": {}
    }

    # ===== INITIAL POPULATION =====
    population = [toolbox.individual() for _ in range(POP_SIZE)]

    log("="*70)
    log("Mario AI - Evolutionary Genetic Programming")
    log("="*70)
    log(f"Seed: {seed}")
    log(f"Generations: {GENERATIONS}")
    log("-"*70)
    log("Evaluating initial population...")

    for idx, ind in enumerate(population):
        ind.fitness.values = (evaluate_gp_individual(ind),)

    best_individual = max(population, key=lambda ind: ind.fitness.values[0])
    best_ever = best_individual.fitness.values[0]

    # ===== EVOLUTION LOOP =====
    for gen in range(GENERATIONS):

        log(f"\n--- GENERATION {gen+1}/{GENERATIONS} ---")

        # Seleção + clone
        selected = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, selected))

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values

        # Mutação
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Avaliação
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = (evaluate_gp_individual(ind),)
            
        best_offspring = max(offspring, key=lambda ind: ind.fitness.values[0])
        best_offspring_fitness = best_offspring.fitness.values[0]

        # ===== ELITISMO =====
        elite = tools.selBest(population, 1)
        population = elite + offspring[:-1]

        # ===== MÉTRICAS (CORRETAS) =====
        fitnesses = [ind.fitness.values[0] for ind in population]

        best_gen = max(population, key=lambda ind: ind.fitness.values[0])
        best_gen_fitness = best_gen.fitness.values[0]

        avg_fitness = float(np.mean(fitnesses))
        std_fitness = float(np.std(fitnesses))
        min_fitness = float(np.min(fitnesses))
        median_fitness = float(np.median(fitnesses))

        # diversidade simples
        diversity = len(set(map(str, population))) / len(population)

        # taxa de mudança
        invalid_ratio = len(invalid_ind) / len(population)

        # atualizar melhor global
        if best_gen_fitness > best_ever:
            best_ever = best_gen_fitness
            best_individual = copy.deepcopy(best_gen)
            log(f">>> NEW BEST EVER: {best_ever:.2f}")
            
            # salvar novo melhor indivíduo
            save_best_individual(
                best_individual,
                toolbox,
                filename_py=run_dir / f"best_ever_gen_{gen+1}.py"
            )

        # ===== LOG =====
        log(
            f"[Gen {gen+1}] "
            f"offspring_best={best_offspring_fitness:.2f} | "
            f"pop_best={best_gen_fitness:.2f} | "
            f"avg={avg_fitness:.2f} | "
            f"median={median_fitness:.2f} | "
            f"min={min_fitness:.2f} | "
            f"std={std_fitness:.2f} | "
            f"diversity={diversity:.2f} | "
            f"invalid_ratio={invalid_ratio:.2f} | "
            f"best_ever={best_ever:.2f}"
        )

        # salvar métricas estruturadas
        experiment_data["generations"].append({
            "gen": gen+1,
            "best": best_gen_fitness,
            "avg": avg_fitness,
            "median": median_fitness,
            "min": min_fitness,
            "std": std_fitness,
            "diversity": diversity,
            "invalid_ratio": invalid_ratio,
            "best_ever": best_ever
        })

    # ===== FINAL =====
    log("\n" + "="*70)
    log(f"Final Best Fitness: {best_ever:.2f}")

    save_best_individual(
        best_individual,
        toolbox,
        filename_py=run_dir / "final_best.py"
    )

    experiment_data["final"] = {
        "fitness": float(best_ever)
    }

    with open(run_dir / "experiment_summary.json", "w") as f:
        json.dump(experiment_data, f, indent=2)

    log(f"\n[LOG] Run salvo em: {run_dir}")
    log("="*70)
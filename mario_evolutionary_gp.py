"""
Mario AI - Evolutionary Genetic Programming
============================================
Implementação de um Algoritmo Evolutivo usando Programação Genética
para evoluir controladores autônomos para Super Mario.

Autor: Bernardo Ventura
Data: 27 de Março de 2026
Repositório: https://github.com/bernardo-ventura/UC-NIAI-mario
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
    Path("data/gp_best_agents").mkdir(parents=True, exist_ok=True)
    output_path = Path("data/gp_best_agents") / filename_py
    with output_path.open("w") as f:
        f.write(full_code)
    print(f"Saved executable code to '{output_path}'")

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION: EVOLUTIONARY ALGORITHM (Currently Random Search)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mario_evolutionary_gp.py <seed>")
        sys.exit(1)

    random.seed(int(sys.argv[1]))

    # Parâmetros evolutivos
    # NUM_ITERATIONS = 50  # Will become GENERATIONS
    GENERATIONS = 3
    POP_SIZE = 100
    CXPB = 0.8  # Crossover probability
    MUTPB = 0.2  # Mutation probability

    # Dados do experimento
    experiment_data = {
        "seed": int(sys.argv[1]),
        "generations": [],
        "final": {}
    }

    population = [toolbox.individual() for _ in range(POP_SIZE)]

    print("="*70)
    print("Mario AI - Evolutionary Genetic Programming")
    print("="*70)
    print(f"Seed: {sys.argv[1]}")
    print(f"Generations: {GENERATIONS} (Evolutionary Algorithm)")
    print("-"*70)
    print("Evaluating initial population...")
    for idx, ind in enumerate(population):
        print(f"  Evaluating individual {idx+1}/{len(population)}")
        ind.fitness.values = (evaluate_gp_individual(ind),)

    best_individual = max(population, key=lambda ind: ind.fitness.values[0])
    best_fitness = best_individual.fitness.values[0]

    for gen in range(GENERATIONS):
        print(f"\n--- GENERATION {gen+1}/{GENERATIONS} ---")

        # Evolução
        selected = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, selected))

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i-1], offspring[i])

        # Mutação
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)

        # Avaliação dos descendentes
        for ind in offspring:
            ind.fitness.values = (evaluate_gp_individual(ind),)

        # Estatísticas da geração
        fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
        avg_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0
        std_fitness = float(np.std(fitnesses)) if fitnesses else 0.0
        best_gen = max(population, key=lambda ind: ind.fitness.values[0])
        best_gen_fitness = best_gen.fitness.values[0]
        print(f"Fitness médio: {avg_fitness:.2f} | Desvio padrão: {std_fitness:.2f} | Melhor: {best_gen_fitness}")

        # Salvar estatísticas
        experiment_data["generations"].append({
            "gen": gen+1,
            "best": best_gen_fitness,
            "avg": avg_fitness,
            "std": std_fitness
        })

        # Salvar melhor indivíduo a cada 5 gerações
        if (gen+1) % 5 == 0:
            save_best_individual(best_gen, toolbox, filename_py=f"mario_best_gen{gen+1}.py")

        # Elitismo
        elite = tools.selBest(population, 1)
        population = elite + offspring[:-1]

        # Atualizar melhor global
        if best_gen.fitness.values[0] > best_fitness:
            best_fitness = best_gen.fitness.values[0]
            best_individual = copy.deepcopy(best_gen)
            print(f">>> New BEST EVER: {best_fitness}")

    print("\n" + "="*70)
    if best_individual:
        print(f"Final Best Fitness: {best_fitness}")
        save_best_individual(best_individual, toolbox)
        # Dados finais (exemplo: fitness, distância, kills)
        experiment_data["final"] = {
            "fitness": float(best_fitness),
            # Placeholder: adicione coleta real de distância/kills se disponível
            "distance": None,
            "kills": None
        }
    else:
        print("No valid programs were found or evaluated.")

    # Salvar JSON do experimento
    Path("data/gp_best_agents").mkdir(parents=True, exist_ok=True)
    json_path = Path("data/gp_best_agents") / f"experiment_seed{sys.argv[1]}.json"
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=2)
    print(f"\n[LOG] Dados do experimento salvos em {json_path}")
    print("="*70)

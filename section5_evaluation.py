"""
Section 5 - Evaluation and Comparative Analysis
===============================================
Compara 5 Evolved Agents vs 1 Random Agent em 3 níveis diferentes.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents import CodeAgent
from tasks import MoveForwardTask
import marioai
from mario_random_search_gp import pset, safe_gen_grow, indent
from deap import creator, base, gp

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

PORT = 4242

# 3 níveis diferentes para testar generalização
LEVELS = [
    {'id': 1, 'seed': 1, 'difficulty': 0, 'type': 0},
    {'id': 2, 'seed': 42, 'difficulty': 1, 'type': 0},
    {'id': 3, 'seed': 100, 'difficulty': 2, 'type': 0},
]

N_TRIALS = 1  # Cada agent joga 1 vez cada nível

# ============================================================================
# TASK CUSTOMIZADA PARA COLETAR KILLS + COINS
# ============================================================================

class EvaluationTask(MoveForwardTask):
    """Task que rastreia kills e coins conforme Paper Section 5.2."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_kills = 0
        self.min_enemies_seen = 99999
    
    def reset(self):
        super().reset()
        self.total_kills = 0
        self.min_enemies_seen = 99999
    
    def compute_reward(self, current_obs, last_obs):
        """Rastreia kills: quando inimigos desaparecem da tela."""
        
        if current_obs.enemies is not None:
            current_enemies = len(current_obs.enemies)
            
            if last_obs is None:
                self.min_enemies_seen = current_enemies
            else:
                # Quando número de inimigos diminui = kills
                if current_enemies < self.min_enemies_seen:
                    kills = self.min_enemies_seen - current_enemies
                    self.total_kills += kills
                    self.min_enemies_seen = current_enemies
        
        return super().compute_reward(current_obs, last_obs)

# ============================================================================
# FUNÇÕES
# ============================================================================

def load_evolved_agents():
    """Carrega os 5 evolved agents das 5 seeds."""
    agents = []
    runs_dir = Path("experiments/move_forward_5_runs")
    
    for agent_file in sorted(runs_dir.glob("*/final_best.py")):
        with open(agent_file, 'r') as f:
            code = f.read()
        agent = CodeAgent()
        agent.action_function = code
        
        # Extrair seed do nome da pasta
        agent_id = agent_file.parent.name
        agents.append({'id': agent_id, 'agent': agent})
    
    return agents[:5]  # Garantir 5 agents


def generate_random_agent():
    """Gera 1 random agent como baseline."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    
    expr = safe_gen_grow(pset, min_=1, max_=3, type_=None)
    individual = creator.Individual(expr)
    code_body = toolbox.compile(individual)
    full_code = f"def corre(action, landscape, enemies, can_jump, on_ground, Mario, Sprite, **kwargs):\n{indent(code_body)}"
    
    agent = CodeAgent()
    agent.action_function = full_code
    
    return {'id': 'random_baseline', 'agent': agent}


def run_trial(agent, task, level_config):
    """
    Roda 1 trial e coleta métricas do Paper Section 5.2:
    - Completion Rate
    - Traversal Efficiency (max distance)
    - Reward Acquisition (kills + coins)
    """
    # Configurar nível
    task.level_seed = level_config['seed']
    task.level_difficulty = level_config['difficulty']
    task.level_type = level_config['type']
    
    # Executar episódio
    exp = marioai.Experiment(task, agent)
    exp.max_fps = -1
    
    try:
        exp.doEpisodes(1)
        
        # Coletar métricas
        completed = 1 if task.status == 1 else 0
        max_distance = task.max_x if hasattr(task, 'max_x') else 0
        kills = task.total_kills
        
        # Coins vem do observation final (FIT packet)
        coins = 0
        if task.last_observation and hasattr(task.last_observation, 'coins'):
            coins = task.last_observation.coins
        
        return {
            'completed': completed,
            'max_distance': max_distance,
            'kills': kills,
            'coins': coins,
            'status': task.status
        }
    except Exception as e:
        print(f" ⚠️ Error: {e}")
        return {
            'completed': 0,
            'max_distance': 0,
            'kills': 0,
            'coins': 0,
            'status': -1
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("📊 SECTION 5 - EVALUATION AND COMPARATIVE ANALYSIS")
    print("="*80)
    print("Evolved Agents: 5")
    print("Random Baseline: 1")
    print(f"Levels: {len(LEVELS)}")
    print(f"Trials per agent per level: {N_TRIALS}")
    print(f"Total runs: 6 agents × {len(LEVELS)} levels × {N_TRIALS} trials = {6 * len(LEVELS) * N_TRIALS}")
    print("="*80)
    
    # 1. Carregar evolved agents
    print("\n1️⃣ Loading evolved agents...")
    evolved_agents = load_evolved_agents()
    print(f"   ✓ Loaded {len(evolved_agents)} evolved agents")
    for ag in evolved_agents:
        print(f"      - {ag['id']}")
    
    # 2. Gerar random agent
    print("\n2️⃣ Generating random baseline agent...")
    random_agent = generate_random_agent()
    print(f"   ✓ Generated {random_agent['id']}")
    
    # 3. Preparar lista de todos os agents
    all_agents = evolved_agents + [random_agent]
    
    # 4. Conectar ao servidor e criar task reutilizável
    print("\n3️⃣ Connecting to Mario server...")
    task = EvaluationTask(visualization=False, port=PORT, init_mario_mode=0)
    print("   ✓ Connected")
    
    # 5. Coletar dados
    print(f"\n4️⃣ Running evaluation ({6 * len(LEVELS) * N_TRIALS} trials)...")
    print()
    
    all_results = []
    total_runs = len(all_agents) * len(LEVELS) * N_TRIALS
    current_run = 0
    
    for agent_info in all_agents:
        agent_type = 'evolved' if agent_info['id'] != 'random_baseline' else 'random'
        
        for level in LEVELS:
            for trial in range(1, N_TRIALS + 1):
                current_run += 1
                
                # Rodar trial (task é reutilizada, apenas muda parâmetros internamente)
                result = run_trial(agent_info['agent'], task, level)
                
                # Salvar resultado
                result_data = {
                    'agent_type': agent_type,
                    'agent_id': agent_info['id'],
                    'level_id': level['id'],
                    'level_seed': level['seed'],
                    'level_difficulty': level['difficulty'],
                    'trial': trial,
                    **result
                }
                all_results.append(result_data)
                
                # Print progress
                status_icon = '✓' if result['completed'] else '✗'
                print(f"   [{current_run:3d}/{total_runs}] {agent_type:7s} {agent_info['id']:35s} | "
                      f"Level {level['id']} | {status_icon} dist={int(result['max_distance']):4d} "
                      f"kills={result['kills']:2d} coins={result['coins']:2d}")
    
    # 5. Salvar resultados brutos
    output_file = Path("data/section5_evaluation.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print("="*80)
    print(f"✅ Evaluation complete!")
    print(f"   Raw data saved to: {output_file}")
    print(f"   Total trials: {len(all_results)}")
    print("="*80)


if __name__ == '__main__':
    main()

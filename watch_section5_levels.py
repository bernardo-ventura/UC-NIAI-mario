"""
Watch Agent Playing Section 5 Levels
=====================================
Visualiza um agent jogando os 3 níveis da avaliação Section 5.

Uso:
    python watch_section5_levels.py <agent_file.py>
    
Exemplos:
    python watch_section5_levels.py experiments/move_forward_5_runs/agent_seed_456/final_best.py
    python watch_section5_levels.py experiments/move_forward_5_runs/agent_seed_123/final_best.py
"""

import sys
from pathlib import Path
from agents import CodeAgent
from tasks import MoveForwardTask
import marioai

# 3 níveis da avaliação Section 5
SECTION5_LEVELS = [
    {'name': 'Level 1', 'seed': 1,   'difficulty': 0, 'type': 0},
    {'name': 'Level 2', 'seed': 42,  'difficulty': 1, 'type': 0},
    {'name': 'Level 3', 'seed': 100, 'difficulty': 2, 'type': 0},
]


def watch_agent_on_level(agent, task, level_config):
    """Roda agent em um nível com visualização."""
    
    # Configurar nível
    task.level_seed = level_config['seed']
    task.level_difficulty = level_config['difficulty']
    task.level_type = level_config['type']
    
    # Habilitar visualização
    task.enable_visualization()
    
    # Criar experimento
    exp = marioai.Experiment(task, agent)
    exp.max_fps = 30  # FPS limitado para visualização confortável
    
    # Rodar episódio
    print(f"   ▶️  Playing... (press Ctrl+C to skip)", flush=True)
    try:
        exp.doEpisodes(1)
    except KeyboardInterrupt:
        print("   ⏭️  Skipped!")
        return False, 0
    
    # Resultado
    won = (task.status == 1)
    max_dist = task.max_x if hasattr(task, 'max_x') else 0
    
    return won, max_dist


def main():
    if len(sys.argv) < 2:
        print("Uso: python watch_section5_levels.py <agent_file.py>")
        print("\nExemplos:")
        print("  python watch_section5_levels.py experiments/move_forward_5_runs/agent_seed_456/final_best.py")
        print("  python watch_section5_levels.py experiments/move_forward_5_runs/agent_seed_123/final_best.py")
        sys.exit(1)
    
    agent_file = Path(sys.argv[1])
    
    if not agent_file.exists():
        print(f"❌ Arquivo não encontrado: {agent_file}")
        sys.exit(1)
    
    print("="*80)
    print("🎮 WATCH: SECTION 5 EVALUATION LEVELS")
    print("="*80)
    print(f"Agent: {agent_file.name}")
    print(f"Path: {agent_file}")
    print(f"Levels: {len(SECTION5_LEVELS)}")
    print("="*80)
    
    # Carregar agente
    print("\n📂 Loading agent...")
    with open(agent_file, 'r') as f:
        code_str = f.read()
    
    agent = CodeAgent()
    agent.action_function = code_str
    print("✓ Agent loaded\n")
    
    # Criar task reutilizável
    print("🔌 Connecting to Mario server...")
    task = MoveForwardTask(
        visualization=True,
        port=4242,
        init_mario_mode=0
    )
    print("✓ Connected\n")
    
    # Jogar cada nível
    print("-"*80)
    print("🎬 PLAYING LEVELS")
    print("-"*80)
    
    results = []
    
    for i, level in enumerate(SECTION5_LEVELS, 1):
        print(f"\n[{i}/3] {level['name']}")
        print(f"      seed={level['seed']}, difficulty={level['difficulty']}, type={level['type']}")
        
        won, distance = watch_agent_on_level(agent, task, level)
        results.append({
            'level': level['name'],
            'won': won,
            'distance': distance
        })
        
        status = "✅ COMPLETED" if won else f"❌ FAILED (reached {distance:.0f}px)"
        print(f"      {status}\n")
    
    # Resultado final
    print("="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    completed = sum(1 for r in results if r['won'])
    
    print(f"\nLevels completed: {completed}/3")
    print("\nDetailed results:")
    for r in results:
        status = "✓" if r['won'] else "✗"
        dist_text = 'COMPLETED' if r['won'] else f"Failed at {r['distance']:.0f}px"
        print(f"  {status} {r['level']:<10} → {dist_text}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

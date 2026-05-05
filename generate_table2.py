"""
Generate Table 2 - Section 5 Comparison
========================================
Processa dados brutos e gera tabela conforme Paper Section 5.3
"""

import json
import csv
import numpy as np
from pathlib import Path

def load_data():
    """Carrega dados brutos."""
    with open('data/section5_evaluation.json', 'r') as f:
        return json.load(f)

def calculate_stats(data):
    """Calcula média e desvio padrão."""
    if len(data) == 0:
        return 0, 0
    
    arr = np.array(data)
    return arr.mean(), arr.std()

def main():
    print("="*80)
    print("📊 TABLE 2 - COMPARISON OF EVOLVED vs RANDOM BASELINE")
    print("="*80)
    
    # Carregar dados
    all_data = load_data()
    
    # Separar evolved vs random
    evolved_data = [d for d in all_data if d['agent_type'] == 'evolved']
    random_data = [d for d in all_data if d['agent_type'] == 'random']
    
    print(f"\nData loaded:")
    print(f"  Evolved trials: {len(evolved_data)}")
    print(f"  Random trials: {len(random_data)}")
    
    # ========================================================================
    # MÉTRICA 1: LEVELS CLEARED (Completion Rate)
    # ========================================================================
    evolved_completed = [d['completed'] for d in evolved_data]
    random_completed = [d['completed'] for d in random_data]
    
    evolved_completion_rate = (sum(evolved_completed) / len(evolved_completed)) * 100
    random_completion_rate = (sum(random_completed) / len(random_completed)) * 100
    
    # STD da taxa de conclusão
    evolved_completion_std = np.std(evolved_completed) * 100
    random_completion_std = np.std(random_completed) * 100
    
    # Improvement
    if random_completion_rate > 0:
        completion_improvement = ((evolved_completion_rate - random_completion_rate) / random_completion_rate) * 100
    else:
        completion_improvement = float('inf') if evolved_completion_rate > 0 else 0
    
    # ========================================================================
    # MÉTRICA 2: MAX DISTANCE (Traversal Efficiency)
    # ========================================================================
    evolved_distances = [d['max_distance'] for d in evolved_data]
    random_distances = [d['max_distance'] for d in random_data]
    
    evolved_dist_mean, evolved_dist_std = calculate_stats(evolved_distances)
    random_dist_mean, random_dist_std = calculate_stats(random_distances)
    
    if random_dist_mean > 0:
        distance_improvement = ((evolved_dist_mean - random_dist_mean) / random_dist_mean) * 100
    else:
        distance_improvement = float('inf') if evolved_dist_mean > 0 else 0
    
    # ========================================================================
    # MÉTRICA 3: REWARD ACQUISITION (Kills + Coins)
    # ========================================================================
    evolved_kills = [d['kills'] for d in evolved_data]
    random_kills = [d['kills'] for d in random_data]
    
    evolved_coins = [d['coins'] for d in evolved_data]
    random_coins = [d['coins'] for d in random_data]
    
    # Reward = kills + coins
    evolved_rewards = [k + c for k, c in zip(evolved_kills, evolved_coins)]
    random_rewards = [k + c for k, c in zip(random_kills, random_coins)]
    
    evolved_reward_mean, evolved_reward_std = calculate_stats(evolved_rewards)
    random_reward_mean, random_reward_std = calculate_stats(random_rewards)
    
    if random_reward_mean > 0:
        reward_improvement = ((evolved_reward_mean - random_reward_mean) / random_reward_mean) * 100
    else:
        reward_improvement = float('inf') if evolved_reward_mean > 0 else 0
    
    # ========================================================================
    # ESTATÍSTICAS DETALHADAS
    # ========================================================================
    
    evolved_kills_mean, evolved_kills_std = calculate_stats(evolved_kills)
    random_kills_mean, random_kills_std = calculate_stats(random_kills)
    
    evolved_coins_mean, evolved_coins_std = calculate_stats(evolved_coins)
    random_coins_mean, random_coins_std = calculate_stats(random_coins)
    
    if random_kills_mean > 0:
        kills_improvement = ((evolved_kills_mean - random_kills_mean) / random_kills_mean) * 100
    else:
        kills_improvement = float('inf') if evolved_kills_mean > 0 else 0
    
    # ========================================================================
    # IMPRIMIR TABELA 2 (formato do paper)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TABLE 2: Comparison of Evolved Controller vs. Random Search Baseline")
    print("="*80)
    print()
    print(f"{'Metric':<25} | {'Random Search':<20} | {'Evolved Agent':<20} | {'Improvement':>12}")
    print(f"{'':25} | {'(Avg ± STD)':<20} | {'(Avg ± STD)':<20} | {'(%)':>12}")
    print("-" * 86)
    
    # Linha 1: Levels Cleared (Completion Rate)
    print(f"{'Levels Cleared (%)':<25} | "
          f"{random_completion_rate:6.1f} ± {random_completion_std:5.1f}%      | "
          f"{evolved_completion_rate:6.1f} ± {evolved_completion_std:5.1f}%      | "
          f"{completion_improvement:>11.1f}%")
    
    # Linha 2: Max Distance
    print(f"{'Max Distance (px)':<25} | "
          f"{random_dist_mean:7.1f} ± {random_dist_std:6.1f}   | "
          f"{evolved_dist_mean:7.1f} ± {evolved_dist_std:6.1f}   | "
          f"{distance_improvement:>11.1f}%")
    
    # Linha 3: Enemies Killed
    print(f"{'Enemies Killed':<25} | "
          f"{random_kills_mean:7.2f} ± {random_kills_std:6.2f}   | "
          f"{evolved_kills_mean:7.2f} ± {evolved_kills_std:6.2f}   | "
          f"{kills_improvement:>11.1f}%")
    
    print("="*86)
    
    # ========================================================================
    # ESTATÍSTICAS ADICIONAIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("ADDITIONAL STATISTICS")
    print("="*80)
    
    print(f"\n📊 Completion Details:")
    print(f"  Evolved: {sum(evolved_completed)}/{len(evolved_completed)} levels completed")
    print(f"  Random:  {sum(random_completed)}/{len(random_completed)} levels completed")
    
    print(f"\n📊 Per-Agent Performance (Evolved):")
    agents = {}
    for d in evolved_data:
        aid = d['agent_id']
        if aid not in agents:
            agents[aid] = []
        agents[aid].append(d['completed'])
    
    for aid, completions in agents.items():
        completed = sum(completions)
        total = len(completions)
        print(f"  {aid:<35} → {completed}/{total} levels completed")
    
    print(f"\n📊 Per-Level Performance:")
    for level_id in [1, 2, 3]:
        evolved_level = [d for d in evolved_data if d['level_id'] == level_id]
        random_level = [d for d in random_data if d['level_id'] == level_id]
        
        evolved_level_completed = sum(d['completed'] for d in evolved_level)
        random_level_completed = sum(d['completed'] for d in random_level)
        
        print(f"  Level {level_id}: Evolved {evolved_level_completed}/{len(evolved_level)} | "
              f"Random {random_level_completed}/{len(random_level)}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)
    
    # ========================================================================
    # SALVAR TABELA EM CSV
    # ========================================================================
    
    csv_file = Path("data/table2_section5.csv")
    csv_file.parent.mkdir(exist_ok=True)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Cabeçalho
        writer.writerow(['Metric', 'Random_Mean', 'Random_STD', 'Evolved_Mean', 'Evolved_STD', 'Improvement_%'])
        
        # Dados
        writer.writerow([
            'Levels_Cleared_%',
            f'{random_completion_rate:.1f}',
            f'{random_completion_std:.1f}',
            f'{evolved_completion_rate:.1f}',
            f'{evolved_completion_std:.1f}',
            'inf' if completion_improvement == float('inf') else f'{completion_improvement:.1f}'
        ])
        
        writer.writerow([
            'Max_Distance_px',
            f'{random_dist_mean:.1f}',
            f'{random_dist_std:.1f}',
            f'{evolved_dist_mean:.1f}',
            f'{evolved_dist_std:.1f}',
            f'{distance_improvement:.1f}'
        ])
        
        writer.writerow([
            'Enemies_Killed',
            f'{random_kills_mean:.2f}',
            f'{random_kills_std:.2f}',
            f'{evolved_kills_mean:.2f}',
            f'{evolved_kills_std:.2f}',
            'inf' if kills_improvement == float('inf') else f'{kills_improvement:.1f}'
        ])
    
    print(f"\n💾 Table 2 saved to: {csv_file}")
    print("="*80)


if __name__ == '__main__':
    main()

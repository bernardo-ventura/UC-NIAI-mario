"""
Fitness Convergence Curve - Section 5
======================================
Plota as curvas de convergência dos 5 agents evolved + random baseline.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_convergence_data():
    """Carrega dados de convergência de cada run."""
    runs_dir = Path("experiments/move_forward_5_runs")
    
    convergence_data = []
    
    for run_dir in sorted(runs_dir.glob("agent_seed_*")):
        summary_file = run_dir / "experiment_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extrair seed do JSON
            seed = data['seed']
            
            # Extrair BEST de cada geração (não best_ever)
            best_fitness = [gen['best'] for gen in data['generations']]
            avg_fitness = [gen['avg'] for gen in data['generations']]
            
            convergence_data.append({
                'seed': seed,
                'name': run_dir.name,
                'n_generations': len(data['generations']),
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'final_best': best_fitness[-1] if best_fitness else 0
            })
    
    return convergence_data

def get_random_baseline():
    """Carrega dados de convergência dos 5 random search baselines."""
    runs_dir = Path("experiments/random_search_baseline")
    
    random_data = []
    
    for run_dir in sorted(runs_dir.glob("random_seed_*")):
        summary_file = run_dir / "experiment_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extrair BEST de cada geração (não best_ever)
            best_fitness = [gen['best'] for gen in data['generations']]
            
            random_data.append({
                'seed': data['seed'],
                'best_fitness': best_fitness,
                'final_best': best_fitness[-1] if best_fitness else 0
            })
    
    return random_data

def main():
    print("="*80)
    print("📈 GENERATING FITNESS CONVERGENCE CURVE")
    print("="*80)
    
    # Carregar dados de convergência
    print("\n📂 Loading convergence data from 5 runs...")
    convergence_data = load_convergence_data()
    
    for run in convergence_data:
        print(f"   ✓ Seed {run['seed']}: {run['n_generations']} generations, final fitness = {run['final_best']:.2f}")
    
    # Calcular baseline do random
    print("\n📊 Loading random search baseline data...")
    random_data = get_random_baseline()
    
    for run in random_data:
        print(f"   ✓ Seed {run['seed']}: 30 generations, final fitness = {run['final_best']:.2f}")
    
    # ========================================================================
    # CRIAR GRÁFICO DE CONVERGÊNCIA
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cores para cada seed (mais vibrantes e distinguíveis)
    colors_evolved = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_random = ['#e74c3c', '#e67e22', '#f39c12', '#16a085', '#8e44ad']  # Cores mais escuras e visíveis
    
    # Plotar curvas de cada evolved agent (linhas sólidas)
    for i, run in enumerate(convergence_data):
        generations = list(range(1, len(run['best_fitness']) + 1))  # Gerações 1-30
        
        print(f"   Evolved seed {run['seed']}: {len(run['best_fitness'])} pontos")
        
        # Curva do best fitness (best_ever acumulado) - TODOS OS 30 PONTOS
        ax.plot(generations, run['best_fitness'], 
                color=colors_evolved[i], 
                linewidth=2.5, 
                label=f"Evolved (seed={run['seed']})",
                marker='o',
                markersize=5,
                alpha=0.9,
                zorder=2)
    
    # Plotar curvas de cada random baseline agent (linhas tracejadas)
    for i, run in enumerate(random_data):
        generations = list(range(1, len(run['best_fitness']) + 1))
        
        print(f"   Random seed {run['seed']}: {len(run['best_fitness'])} pontos")
        
        # TODOS OS 30 PONTOS visíveis
        ax.plot(generations, run['best_fitness'], 
                color=colors_random[i], 
                linewidth=2.5, 
                label=f"Random Search (seed={run['seed']})",
                linestyle='--',
                marker='s',  # Quadrado para diferenciar
                markersize=5,
                alpha=0.85,
                zorder=1)
    
    # Configurações do gráfico
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Fitness (Max Distance)', fontsize=12, fontweight='bold')
    ax.set_title('Fitness Convergence Curve: Evolved Agents vs Random Search Baseline\n(5 Independent Runs per Strategy, N=5)',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Melhorar aparência
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ajustar limites do eixo Y para incluir TODOS os valores (inclusive negativos)
    # Não forçar bottom=0, deixar o matplotlib ajustar automaticamente
    # ax.set_ylim(bottom=None)  # Remove limite inferior fixo
    
    plt.tight_layout()
    
    # Salvar
    output_png = Path('data/fitness_convergence.png')
    output_jpg = Path('data/fitness_convergence.jpg')
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_jpg, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Convergence curve saved:")
    print(f"   PNG: {output_png}")
    print(f"   JPG: {output_jpg}")
    print(f"   Resolution: 300 DPI")
    
    plt.close()
    
    # ========================================================================
    # ESTATÍSTICAS FINAIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 CONVERGENCE STATISTICS")
    print("="*80)
    
    final_fitness_evolved = [run['final_best'] for run in convergence_data]
    final_fitness_random = [run['final_best'] for run in random_data]
    
    print(f"\nEvolved Agents (final generation):")
    print(f"  Mean fitness: {np.mean(final_fitness_evolved):.2f}")
    print(f"  Std fitness:  {np.std(final_fitness_evolved):.2f}")
    print(f"  Min fitness:  {np.min(final_fitness_evolved):.2f}")
    print(f"  Max fitness:  {np.max(final_fitness_evolved):.2f}")
    
    print(f"\nRandom Search Baseline (final generation):")
    print(f"  Mean fitness: {np.mean(final_fitness_random):.2f}")
    print(f"  Std fitness:  {np.std(final_fitness_random):.2f}")
    print(f"  Min fitness:  {np.min(final_fitness_random):.2f}")
    print(f"  Max fitness:  {np.max(final_fitness_random):.2f}")
    
    improvement = ((np.mean(final_fitness_evolved) - np.mean(final_fitness_random)) / abs(np.mean(final_fitness_random))) * 100
    print(f"\nImprovement: +{improvement:.1f}%")
    
    print("\n" + "="*80)
    print("🎨 Fitness convergence visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()

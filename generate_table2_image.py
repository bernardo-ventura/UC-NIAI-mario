"""
Generate Table 2 - Visual Image
================================
Gera uma imagem bonita da Tabela 2 para documentação.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
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
    print("📊 GENERATING TABLE 2 IMAGE")
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
    # CALCULAR MÉTRICAS
    # ========================================================================
    
    # MÉTRICA 1: LEVELS CLEARED
    evolved_completed = [d['completed'] for d in evolved_data]
    random_completed = [d['completed'] for d in random_data]
    
    evolved_completion_rate = (sum(evolved_completed) / len(evolved_completed)) * 100
    random_completion_rate = (sum(random_completed) / len(random_completed)) * 100
    
    evolved_completion_std = np.std(evolved_completed) * 100
    random_completion_std = np.std(random_completed) * 100
    
    if random_completion_rate > 0:
        completion_improvement = ((evolved_completion_rate - random_completion_rate) / random_completion_rate) * 100
    else:
        completion_improvement = float('inf')
    
    # MÉTRICA 2: MAX DISTANCE
    evolved_distances = [d['max_distance'] for d in evolved_data]
    random_distances = [d['max_distance'] for d in random_data]
    
    evolved_dist_mean, evolved_dist_std = calculate_stats(evolved_distances)
    random_dist_mean, random_dist_std = calculate_stats(random_distances)
    
    if random_dist_mean > 0:
        distance_improvement = ((evolved_dist_mean - random_dist_mean) / random_dist_mean) * 100
    else:
        distance_improvement = float('inf')
    
    # MÉTRICA 3: ENEMIES KILLED
    evolved_kills = [d['kills'] for d in evolved_data]
    random_kills = [d['kills'] for d in random_data]
    
    evolved_kills_mean, evolved_kills_std = calculate_stats(evolved_kills)
    random_kills_mean, random_kills_std = calculate_stats(random_kills)
    
    if random_kills_mean > 0:
        kills_improvement = ((evolved_kills_mean - random_kills_mean) / random_kills_mean) * 100
    else:
        kills_improvement = float('inf')
    
    # ========================================================================
    # CRIAR TABELA VISUAL
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Dados da tabela
    table_data = [
        ['Metric', 'Random Search\n(Avg ± STD)', 'Evolved Agent\n(Avg ± STD)', 'Improvement\n(%)'],
        [
            'Levels Cleared (%)',
            f'{random_completion_rate:.1f} ± {random_completion_std:.1f}',
            f'{evolved_completion_rate:.1f} ± {evolved_completion_std:.1f}',
            '∞' if completion_improvement == float('inf') else f'+{completion_improvement:.1f}%'
        ],
        [
            'Max Distance (px)',
            f'{random_dist_mean:.1f} ± {random_dist_std:.1f}',
            f'{evolved_dist_mean:.1f} ± {evolved_dist_std:.1f}',
            f'+{distance_improvement:.1f}%'
        ],
        [
            'Enemies Killed',
            f'{random_kills_mean:.2f} ± {random_kills_std:.2f}',
            f'{evolved_kills_mean:.2f} ± {evolved_kills_std:.2f}',
            '∞' if kills_improvement == float('inf') else f'+{kills_improvement:.1f}%'
        ]
    ]
    
    # Criar tabela
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Estilizar
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Cabeçalho
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Linhas alternadas
    for i in range(1, 4):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Coluna de improvement em verde
            if j == 3:
                cell.set_text_props(weight='bold', color='#00AA00')
    
    # Título
    plt.title('Table 2: Comparison of Evolved Controller vs. Random Search Baseline\n(Section 5.3 - Statistical Significance)',
              fontsize=14, fontweight='bold', pad=20)
    
    # Salvar
    output_file = Path('data/table2_section5.png')
    output_file.parent.mkdir(exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Table image saved to: {output_file}")
    print(f"   Resolution: 300 DPI (high quality)")
    print(f"   Format: PNG")
    
    plt.close()
    
    print("\n" + "="*80)
    print("🎨 Visual table generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()

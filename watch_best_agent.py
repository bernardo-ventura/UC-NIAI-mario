"""
Watch Best Agent - Visualiza o melhor agente jogando
=====================================================
Roda o melhor agente salvo COM visualização gráfica para você assistir!

Uso:
    python watch_best_agent.py
    python watch_best_agent.py data/gp_best_agents/mario_best_evolved.py
"""

import sys
import marioai
from agents import CodeAgent
from tasks import MoveForwardTask

def watch_agent(agent_file="data/gp_best_agents/mario_best_evolved.py", num_episodes=1):
    """
    Carrega e roda um agente com visualização gráfica.
    
    Args:
        agent_file: Caminho para o arquivo .py do agente
        num_episodes: Quantos episódios rodar (padrão: 1 para evitar bugs do servidor)
    """
    print("="*70)
    print("🎮 Mario AI - Modo Visualização")
    print("="*70)
    print(f"Carregando agente: {agent_file}")
    print(f"Episódios: {num_episodes}")
    print("-"*70)
    
    # Carregar código do agente
    with open(agent_file, 'r') as f:
        code_str = f.read()
    
    # Criar agente
    agent = CodeAgent()
    agent.action_function = code_str
    
    # Criar task COM VISUALIZAÇÃO! 🎮 (mesma config do random_agent.py)
    task = MoveForwardTask(visualization=True)
    
    # Criar experimento
    exp = marioai.Experiment(task, agent)
    exp.max_fps = 30  # Sem limite de FPS (igual random_agent.py)
    
    print("\n🎬 Iniciando visualização...")
    print("💡 Dica: Feche a janela do jogo para parar\n")
    
    # Rodar episódios
    for episode in range(num_episodes):
        print(f"\n--- Episódio {episode + 1}/{num_episodes} ---")
        
        task.reset()
        rewards = exp.doEpisodes(1)
        
        status_msg = {
            -1: "❌ PERDEU (morreu)",
            0: "⏱️ TIMEOUT (acabou tempo)", 
            1: "🏆 GANHOU!"
        }
        
        print(f"Status: {status_msg.get(task.status, 'Desconhecido')}")
        
        # Mostrar distância apenas se disponível
        if hasattr(agent, 'mario_floats') and agent.mario_floats is not None:
            print(f"Distância percorrida: {agent.mario_floats[0]:.1f}")
        
        print(f"Reward total: {task.cum_reward:.2f}")
    
    print("\n" + "="*70)
    print("✅ Visualização concluída!")
    print("="*70)

if __name__ == "__main__":
    # Permitir especificar arquivo por argumento
    agent_file = sys.argv[1] if len(sys.argv) > 1 else "data/gp_best_agents/mario_best_evolved.py"
    
    # Permitir especificar número de episódios
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    try:
        watch_agent(agent_file, num_episodes)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo não encontrado: {agent_file}")
        print("\nUso:")
        print("  python watch_best_agent.py")
        print("  python watch_best_agent.py data/gp_best_agents/mario_best_evolved.py")
        print("  python watch_best_agent.py data/gp_best_agents/mario_best_evolved.py 5")
    except KeyboardInterrupt:
        print("\n\n⏹️  Visualização interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

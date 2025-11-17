using System;
using System.Collections.Generic;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Implementa o otimizador Adam.
    /// ✅ VERSÃO CORRIGIDA (DISK-BACKED): Esta versão não retém tensores na memória RAM.
    /// Os estados de momento (m, v) são serializados em disco e carregados sob demanda
    /// a cada atualização, garantindo um uso de memória estático e constante.
    /// </summary>
    public class AdamOptimizer
    {
        private readonly float _learningRate;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;

        // Gerenciador de tensores em disco
        private readonly IndividualFileTensorManager _tensorManager;
        
        // Dicionários agora armazenam IDs (strings), não os tensores completos
        private readonly Dictionary<int, string> _m_ids;
        private readonly Dictionary<int, string> _v_ids;
        private readonly Dictionary<int, int> _t;

        /// <summary>
        /// Construtor do otimizador Adam disk-backed.
        /// </summary>
        /// <param name="tensorManager">O gerenciador que controla o armazenamento de tensores em disco.</param>
        public AdamOptimizer(IndividualFileTensorManager tensorManager, float learningRate = 0.001f, float beta1 = 0.9f,
            float beta2 = 0.999f, float epsilon = 1e-7f)
        {
            _tensorManager = tensorManager ?? throw new ArgumentNullException(nameof(tensorManager));
            _learningRate = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;

            _m_ids = new Dictionary<int, string>();
            _v_ids = new Dictionary<int, string>();
            _t = new Dictionary<int, int>();
        }

        /// <summary>
        /// Atualiza os parâmetros de uma camada. Carrega os estados m e v do disco,
        /// executa a atualização na GPU e salva os estados atualizados de volta no disco.
        /// Nenhum tensor de estado permanece na memória após a conclusão do método.
        /// </summary>
        public void UpdateParametersGpu(int layerId, IMathTensor parameters, IMathTensor gradients, IMathEngine mathEngine)
        {
            // O TensorScope garante que os tensores m e v carregados do disco
            // sejam liberados da memória da GPU assim que a atualização terminar.
            using (var scope = new TensorScope($"AdamUpdate_{layerId}", mathEngine, _tensorManager))
            {
                IMathTensor m, v;

                // Etapa 1: Inicializa os arquivos de tensor de momento (m e v) no disco se for a primeira vez.
                if (!_m_ids.ContainsKey(layerId))
                {
                    _m_ids[layerId] = _tensorManager.CreateAndStoreZeros(parameters.Shape, $"Adam_m_layer{layerId}");
                    _v_ids[layerId] = _tensorManager.CreateAndStoreZeros(parameters.Shape, $"Adam_v_layer{layerId}");
                    _t[layerId] = 0;
                }

                // Etapa 2: Carrega os tensores de momento do disco para a memória (CPU ou GPU).
                m = scope.LoadTensor(_m_ids[layerId]);
                v = scope.LoadTensor(_v_ids[layerId]);

                // Etapa 3: Incrementa o timestep.
                _t[layerId]++;
                int t = _t[layerId];

                // Etapa 4: Delega a lógica matemática para a GPU.
                // Esta operação modifica os tensores 'parameters', 'm' e 'v' na VRAM.
                mathEngine.AdamUpdate(parameters, gradients, m, v, _learningRate, _beta1, _beta2, _epsilon, t);

                // Etapa 5: CRÍTICO - Salva os tensores de momento 'm' e 'v' atualizados de volta para o disco.
                // A função OverwriteTensor é mais rápida pois não precisa ler os dados antigos.
                _tensorManager.OverwriteTensor(_m_ids[layerId], m);
                _tensorManager.OverwriteTensor(_v_ids[layerId], v);
            }
            // Ao sair do escopo 'using', os tensores 'm' e 'v' que foram carregados na memória são automaticamente descartados.
        }

        /// <summary>
        /// CRÍTICO: Limpa os estados internos do otimizador.
        /// Este método agora deleta os arquivos de tensor do disco e limpa os dicionários de IDs.
        /// </summary>
        public void Reset()
        {
            Console.WriteLine($"[AdamOptimizer] Limpando {_m_ids.Count} arquivos de estado 'm' e 'v' do disco...");
            int deleteCount = 0;
            
            // Deleta os arquivos de tensor do disco
            foreach (var id in _m_ids.Values)
            {
                _tensorManager.DeleteTensor(id);
                deleteCount++;
            }
            foreach (var id in _v_ids.Values)
            {
                _tensorManager.DeleteTensor(id);
                deleteCount++;
            }

            // Limpa os dicionários de IDs em memória
            _m_ids.Clear();
            _v_ids.Clear();
            _t.Clear();
            
            Console.WriteLine($"[AdamOptimizer] Reset concluído: {deleteCount} arquivos de tensor de estado liberados.");
        }

        /// <summary>
        /// Retorna um conjunto com os IDs de todos os tensores de estado do otimizador (m e v).
        /// Usado para evitar que a limpeza de tensores temporários os delete por engano.
        /// </summary>
        public HashSet<string> GetStateTensorIds()
        {
            var ids = new HashSet<string>();
            foreach (var id in _m_ids.Values)
            {
                ids.Add(id);
            }
            foreach (var id in _v_ids.Values)
            {
                ids.Add(id);
            }
            return ids;
        }
    }
}
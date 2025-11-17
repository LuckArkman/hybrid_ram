using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Estende a rede LSTM base (disk-backed) com a capacidade de gerar texto.
    /// </summary>
    public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
    {
        public readonly VocabularyManager vocabularyManager;
        private readonly ISearchService searchService;
        private readonly int _embeddingSize;

        /// <summary>
        /// Construtor para criar um novo modelo generativo para treinamento.
        /// </summary>
        public GenerativeNeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, string datasetPath,
            ISearchService? searchService, IMathEngine mathEngine)
            : base(vocabSize, embeddingSize, hiddenSize, vocabSize, mathEngine)
        {
            this.vocabularyManager = new VocabularyManager();
            this.searchService = searchService ?? new MockSearchService();
            this._embeddingSize = embeddingSize;

            int loadedVocabSize = vocabularyManager.BuildVocabulary(datasetPath, maxVocabSize: vocabSize);
            if (loadedVocabSize == 0)
            {
                throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
            }
        }

        /// <summary>
        /// Construtor privado para "envolver" um modelo base já carregado.
        /// </summary>
        private GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM baseModel,
            VocabularyManager vocabManager, ISearchService? searchService)
            : base(baseModel) // Chama construtor protegido de cópia
        {
            if (baseModel == null)
                throw new ArgumentNullException(nameof(baseModel), "Modelo base não pode ser nulo");

            this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
            this.searchService = searchService ?? new MockSearchService();

            if (_tensorManager == null || string.IsNullOrEmpty(_weightsEmbeddingId))
            {
                throw new InvalidOperationException("Modelo base está em estado inválido.");
            }

            try
            {
                var shape = _tensorManager.GetShape(_weightsEmbeddingId);
                if (shape == null || shape.Length < 2)
                {
                    throw new InvalidOperationException($"Shape do embedding inválido: {(shape == null ? "null" : $"[{string.Join(", ", shape)}]")}");
                }
                this._embeddingSize = shape[1];
                if (this._embeddingSize <= 0)
                {
                    throw new InvalidOperationException($"Tamanho de embedding inválido: {this._embeddingSize}");
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Falha ao inicializar GenerativeNeuralNetworkLSTM: {ex.Message}", ex);
            }
            Console.WriteLine($"[GenerativeNeuralNetworkLSTM] Inicializado com embedding size: {_embeddingSize}");
        }

        /// <summary>
        /// Método de fábrica estático para carregar um modelo e envolvê-lo.
        /// </summary>
        public static GenerativeNeuralNetworkLSTM? Load(string modelPath, IMathEngine mathEngine,
            VocabularyManager vocabManager, ISearchService? searchService)
        {
            var baseModel = NeuralNetworkLSTM.LoadModel(modelPath, mathEngine);
            if (baseModel == null)
            {
                return null;
            }
            return new GenerativeNeuralNetworkLSTM(baseModel, vocabManager, searchService);
        }

        /// <summary>
        /// Gera uma continuação de texto a partir de um prompt.
        /// NOTA: Esta é uma implementação simplificada para fins de demonstração.
        /// </summary>
        public string GenerateResponse(string inputText, int maxLength = 50)
        {
             if (string.IsNullOrEmpty(inputText)) return "Erro: Entrada vazia ou nula.";
             // A implementação completa exigiria um 'ForwardPass' token a token,
             // o que é complexo na arquitetura ZeroRAM. Esta função serve como placeholder.
             Console.WriteLine("[GenerateResponse] A geração de texto token-a-token na arquitetura ZeroRAM é complexa e não está totalmente implementada para inferência.");
             return "Geração de resposta não implementada nesta fase.";
        }

        /// <summary>
        /// Calcula a perda para uma sequência para fins de validação.
        /// </summary>
        public float CalculateSequenceLoss(int[] inputIndices, int[] targetIndices)
        {
            using (var masterScope = new TensorScope("CalculateLoss", _mathEngine, _tensorManager))
            {
                var weights = new ModelWeights
                {
                    Embedding = masterScope.LoadTensor(_weightsEmbeddingId), W_if = masterScope.LoadTensor(_weightsInputForgetId), W_hf = masterScope.LoadTensor(_weightsHiddenForgetId), B_f = masterScope.LoadTensor(_biasForgetId),
                    W_ii = masterScope.LoadTensor(_weightsInputInputId), W_hi = masterScope.LoadTensor(_weightsHiddenInputId), B_i = masterScope.LoadTensor(_biasInputId),
                    W_ic = masterScope.LoadTensor(_weightsInputCellId), W_hc = masterScope.LoadTensor(_weightsHiddenCellId), B_c = masterScope.LoadTensor(_biasCellId),
                    W_io = masterScope.LoadTensor(_weightsInputOutputId), W_ho = masterScope.LoadTensor(_weightsHiddenOutputId), B_o = masterScope.LoadTensor(_biasOutputId),
                    W_hy = masterScope.LoadTensor(_weightsHiddenOutputFinalId), B_y = masterScope.LoadTensor(_biasOutputFinalId),
                    LN_f_gamma = masterScope.LoadTensor(_lnForgetGammaId), LN_f_beta = masterScope.LoadTensor(_lnForgetBetaId), LN_i_gamma = masterScope.LoadTensor(_lnInputGammaId), LN_i_beta = masterScope.LoadTensor(_lnInputBetaId),
                    LN_c_gamma = masterScope.LoadTensor(_lnCellGammaId), LN_c_beta = masterScope.LoadTensor(_lnCellBetaId), LN_o_gamma = masterScope.LoadTensor(_lnOutputGammaId), LN_o_beta = masterScope.LoadTensor(_lnOutputBetaId)
                };

                var (loss, swapFiles) = base.ForwardPassZeroRAM(inputIndices, targetIndices, weights);

                // A limpeza dos swap files é crucial para não acumular lixo no disco.
                foreach (var file in swapFiles)
                {
                    _swapManager.DeleteSwapFile(file);
                }
                
                return loss;
            }
        }

        public void Reset()
        {
            base.ResetHiddenState();
        }

        private int GetTokenIndex(string token)
        {
            return vocabularyManager.Vocab.TryGetValue(token.ToLower(), out int tokenIndex)
                ? tokenIndex
                : vocabularyManager.Vocab["<UNK>"];
        }

        private string[] Tokenize(string text)
        {
            return text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        }

        private int SampleToken(Tensor output)
        {
            float[] probs = output.GetData();
            float r = (float)new Random().NextDouble();
            float cumulative = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (r <= cumulative) return i;
            }
            return probs.Length - 1;
        }

/// <summary>
/// 🔥 EXECUTA UMA VERIFICAÇÃO DE SANIDADE COMPLETA (ZERO-RAM) (VERSÃO CORRIGIDA)
/// Roda um ciclo completo de forward, backward e update em dados sintéticos
/// para garantir que a arquitetura está funcional antes de iniciar o treinamento real.
/// Lança uma exceção se qualquer etapa crítica falhar.
/// </summary>
public void RunSanityCheckZeroRAM()
{
    Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║        🚀 INICIANDO VERIFICAÇÃO DE SANIDADE (ZERO-RAM)     ║");
    Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");

    var inputIndices = new[] { 5, 10 };
    var targetIndices = new[] { 10, 15 };
    Console.WriteLine($"[Sanity Check] Usando dados sintéticos: Input={{{string.Join(",", inputIndices)}}}, Target={{{string.Join(",", targetIndices)}}}");

    List<string> forwardSwapFiles = null;
    Dictionary<string, string> gradIds = null;

    try
    {
        using (var masterScope = new TensorScope("SanityCheckMaster", _mathEngine, _tensorManager))
        {
            var weights = new ModelWeights {
                Embedding = masterScope.LoadTensor(_weightsEmbeddingId), W_if = masterScope.LoadTensor(_weightsInputForgetId), W_hf = masterScope.LoadTensor(_weightsHiddenForgetId), B_f = masterScope.LoadTensor(_biasForgetId),
                W_ii = masterScope.LoadTensor(_weightsInputInputId), W_hi = masterScope.LoadTensor(_weightsHiddenInputId), B_i = masterScope.LoadTensor(_biasInputId),
                W_ic = masterScope.LoadTensor(_weightsInputCellId), W_hc = masterScope.LoadTensor(_weightsHiddenCellId), B_c = masterScope.LoadTensor(_biasCellId),
                W_io = masterScope.LoadTensor(_weightsInputOutputId), W_ho = masterScope.LoadTensor(_weightsHiddenOutputId), B_o = masterScope.LoadTensor(_biasOutputId),
                W_hy = masterScope.LoadTensor(_weightsHiddenOutputFinalId), B_y = masterScope.LoadTensor(_biasOutputFinalId),
                LN_f_gamma = masterScope.LoadTensor(_lnForgetGammaId), LN_f_beta = masterScope.LoadTensor(_lnForgetBetaId), LN_i_gamma = masterScope.LoadTensor(_lnInputGammaId), LN_i_beta = masterScope.LoadTensor(_lnInputBetaId),
                LN_c_gamma = masterScope.LoadTensor(_lnCellGammaId), LN_c_beta = masterScope.LoadTensor(_lnCellBetaId), LN_o_gamma = masterScope.LoadTensor(_lnOutputGammaId), LN_o_beta = masterScope.LoadTensor(_lnOutputBetaId)
            };
            
            Console.WriteLine("\n[Sanity Check] Fase 1/3: Executando Forward Pass...");
            var (loss, swapFiles) = ForwardPassZeroRAM(inputIndices, targetIndices, weights);
            forwardSwapFiles = swapFiles; 
            Console.WriteLine($"[Sanity Check] Forward Pass concluído. Perda inicial: {loss:F4}");

            if (float.IsNaN(loss) || float.IsInfinity(loss)) throw new InvalidOperationException($"Falha na verificação: A perda inicial é {loss}.");
            float expectedLoss = MathF.Log(this.outputSize);
            Console.WriteLine($"[Sanity Check] Perda esperada (aleatória): ~{expectedLoss:F4}");
            if (Math.Abs(loss - expectedLoss) > expectedLoss)
            {
                 Console.ForegroundColor = ConsoleColor.Yellow;
                 Console.WriteLine($"[Sanity Check] AVISO: A perda inicial está mais distante do que o esperado.");
                 Console.ResetColor();
            }

            Console.WriteLine("\n[Sanity Check] Fase 2/3: Executando Backward Pass...");
            gradIds = BackwardPassZeroRAM(inputIndices, targetIndices, forwardSwapFiles, weights);
            Console.WriteLine($"[Sanity Check] Backward Pass concluído. {gradIds.Count} arquivos de gradiente gerados.");

            double totalGradSum = 0;
            foreach (var gradId in gradIds.Values)
            {
                using var gradScope = new TensorScope("GradCheck", _mathEngine, _tensorManager);
                var gradTensor = gradScope.LoadTensor(gradId);
                using var gradCpu = gradTensor.ToCpuTensor();
                foreach (var val in gradCpu.GetData())
                {
                    if (float.IsNaN(val) || float.IsInfinity(val)) throw new InvalidOperationException($"Falha na verificação: Gradiente {Path.GetFileName(gradId)} contém valor inválido ({val}).");
                    totalGradSum += Math.Abs(val);
                }
            }
            Console.WriteLine($"[Sanity Check] Soma absoluta de todos os gradientes: {totalGradSum:E2}");
            if (totalGradSum < 1e-9) throw new InvalidOperationException("Falha na verificação: A soma dos gradientes é próxima de zero.");

            Console.WriteLine("\n[Sanity Check] Fase 3/3: Executando Update Pass (Adam)...");
            var weightIds = new Dictionary<string, string> { { "W_hy", _weightsHiddenOutputFinalId } };
            UpdateAdamGPUPassZeroRAM(weightIds, gradIds);
            Console.WriteLine("[Sanity Check] Update Pass concluído.");
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         ✅ VERIFICAÇÃO DE SANIDADE CONCLUÍDA COM SUCESSO!      ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝\n");
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         ❌ FALHA NA VERIFICAÇÃO DE SANIDADE!               ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
        Console.WriteLine($"[Sanity Check] ERRO: {ex.Message}");
        Console.ResetColor();
        throw; 
    }
    finally
    {
        // ✅ CORREÇÃO: A limpeza foi movida para DENTRO do 'finally', garantindo
        // que ela sempre execute, mas APÓS o try/catch ter sido concluído.
        Console.WriteLine("\n[Sanity Check] Executando limpeza de recursos...");
        if (forwardSwapFiles != null)
        {
            foreach (var file in forwardSwapFiles) _swapManager.DeleteSwapFile(file);
        }
        if (gradIds != null)
        {
            foreach (var gradFileId in gradIds.Values) _tensorManager.DeleteTensor(gradFileId);
        }
        _swapManager.ClearAllSwap();
        Console.WriteLine("[Sanity Check] Limpeza concluída.");
    }
}
    }
}
using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces; // Necessário para IMathEngine e IMathTensor
using System;
using System.IO;
using System.Linq;

namespace Galileu.Node.Services;

public class PrimingService
{
    private readonly string _promptFilePath;

    public PrimingService()
    {
        _promptFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "priming_prompt.txt");
    }

    /// <summary>
    /// Processa o prompt de diretiva para "aquecer" o estado oculto do modelo,
    /// usando a arquitetura ZeroRAM.
    /// </summary>
    /// <param name="model">A instância do modelo a ser preparada.</param>
    public void PrimeModel(GenerativeNeuralNetworkLSTM model)
    {
        if (!File.Exists(_promptFilePath))
        {
            Console.WriteLine($"[PrimingService] Aviso: Arquivo de prompt não encontrado em '{_promptFilePath}'. O modelo não será inicializado.");
            return;
        }

        Console.WriteLine("[PrimingService] Inicializando o modelo com a diretiva de comportamento...");

        var promptText = File.ReadAllText(_promptFilePath);
        if (string.IsNullOrWhiteSpace(promptText))
        {
            Console.WriteLine("[PrimingService] Aviso: Arquivo de prompt está vazio.");
            return;
        }

        var vocabManager = model.vocabularyManager;

        // 1. Tokenizar e converter todo o prompt para índices numéricos
        var tokens = promptText.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var inputIndices = tokens
            .Select(token => vocabManager.Vocab.TryGetValue(token, out int index) ? index : vocabManager.Vocab["<UNK>"])
            .ToArray();

        // Criar um array de "alvos" falso, pois ForwardPassZeroRAM exige. O conteúdo não importa.
        var targetIndices = new int[inputIndices.Length]; 

        // 2. Executar o ForwardPassZeroRAM para processar a sequência de priming
        // A perda retornada é ignorada, pois o objetivo é apenas atualizar os estados h e c.
        try
        {
            // O forward pass atualizará os estados oculto e de célula e os salvará de volta no disco.
            // Usamos um escopo para carregar os pesos necessários para a passagem.
            using (var scope = new TensorScope("Priming", model.GetMathEngine(), model.GetTensorManager()))
            {
                var weights = new ModelWeights {
                    Embedding = scope.LoadTensor(model.GetWeightsEmbeddingId()),
                    W_if = scope.LoadTensor(model.GetWeightsInputForgetId()), W_hf = scope.LoadTensor(model.GetWeightsHiddenForgetId()), B_f = scope.LoadTensor(model.GetBiasForgetId()),
                    W_ii = scope.LoadTensor(model.GetWeightsInputInputId()), W_hi = scope.LoadTensor(model.GetWeightsHiddenInputId()), B_i = scope.LoadTensor(model.GetBiasInputId()),
                    W_ic = scope.LoadTensor(model.GetWeightsInputCellId()), W_hc = scope.LoadTensor(model.GetWeightsHiddenCellId()), B_c = scope.LoadTensor(model.GetBiasCellId()),
                    W_io = scope.LoadTensor(model.GetWeightsInputOutputId()), W_ho = scope.LoadTensor(model.GetWeightsHiddenOutputId()), B_o = scope.LoadTensor(model.GetBiasOutputId()),
                    W_hy = scope.LoadTensor(model.GetWeightsHiddenOutputFinalId()), B_y = scope.LoadTensor(model.GetBiasOutputFinalId()),
                    LN_f_gamma = scope.LoadTensor(model.GetLnForgetGammaId()), LN_f_beta = scope.LoadTensor(model.GetLnForgetBetaId()), LN_i_gamma = scope.LoadTensor(model.GetLnInputGammaId()), LN_i_beta = scope.LoadTensor(model.GetLnInputBetaId()),
                    LN_c_gamma = scope.LoadTensor(model.GetLnCellGammaId()), LN_c_beta = scope.LoadTensor(model.GetLnCellBetaId()), LN_o_gamma = scope.LoadTensor(model.GetLnOutputGammaId()), LN_o_beta = scope.LoadTensor(model.GetLnOutputBetaId())
                };

                // O método ForwardPassZeroRAM é protegido, então precisamos de um wrapper público ou de reflexão.
                // A maneira mais limpa é adicionar um método público na classe base para este propósito específico.
                var (_, swapFiles) = model.RunForwardPassForInference(inputIndices, targetIndices, weights);
                
                // 3. Limpar os arquivos de swap gerados durante o priming
                model.GetSwapManager().ClearAllSwap();
            }

            Console.WriteLine("[PrimingService] Modelo inicializado com sucesso.");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[PrimingService] ERRO CRÍTICO durante a inicialização do modelo: {ex.Message}");
            Console.ResetColor();
        }
    }
}
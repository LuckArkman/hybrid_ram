using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using Galileu.Node.Services;

namespace Galileu.Node.Brain
{
    public class ModelTrainerLSTM
    {
        private readonly IMathEngine _mathEngine;
        private readonly Stopwatch _stopwatch = new Stopwatch();
        private readonly Stopwatch _loteswatch = new Stopwatch();
        private readonly Process _currentProcess;
        private readonly ISearchService _searchService;
        private readonly GpuSyncGuard? _syncGuard; // ✅ NOVO
        private const long CRITICAL_MEMORY_MB = 2500; // ✅ NOVO: Threshold crítico
        private long _peakMemoryUsageMB = 0;
        private GpuMemoryTracker _gpuMemTracker;
        private readonly string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");

        // ✅ NOVO: Parâmetros do Memory Watchdog
        private const long MEMORY_TRIM_THRESHOLD_MB = 2000; // Aciona a ~20 GB de RAM. Ajuste para o seu servidor.
        private long _lastTrimMemory = 0;

        public ModelTrainerLSTM(IMathEngine mathEngine)
        {
            _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            _currentProcess = Process.GetCurrentProcess();
            _searchService = new MockSearchService();
            if (mathEngine is GpuMathEngine gpuEngine)
            {
                _syncGuard = gpuEngine.GetSyncGuard();
            }
        }

        public GenerativeNeuralNetworkLSTM? TrainModel(
            GenerativeNeuralNetworkLSTM initialModel,
            string datasetPath,
            string finalModelPath,
            float learningRate,
            int epochs,
            int batchSize,
            int contextWindowSize,
            float validationSplit)
        {
            int failedBatches = 0;
            if (!File.Exists(datasetPath))
                throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);

            var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "batches.bts");
            using (var datasetService = new DatasetService(swapFilePath))
            {
                datasetService.InitializeAndSplit(File.ReadAllText(datasetPath), contextWindowSize,
                    initialModel.vocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

                Console.WriteLine($"\n[Trainer] Configuração do Ciclo de Treinamento com Liberação Total de Memória.");
                Console.WriteLine($"[Trainer] Memory Watchdog ativado: Limite de ~{MEMORY_TRIM_THRESHOLD_MB} MB.");

                GenerativeNeuralNetworkLSTM? currentModel = initialModel;
                TimeSpan totalElapsedTime = TimeSpan.Zero;
                var trainBatchIndices = datasetService.GetTrainBatchOffsets();
                var validationBatchIndices = datasetService.GetValidationBatchOffsets();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    _stopwatch.Restart();
                    Console.WriteLine($"\n{'═',60}");
                    Console.WriteLine(
                        $"ÉPOCA {epoch + 1}/{epochs} >> Learning Rate: {learningRate} >> {DateTime.UtcNow}");
                    if (currentModel != null) currentModel.warmupSteps = 50;
                    Console.WriteLine($"{'═',60}");

                    double totalEpochLoss = 0;
                    int batchCount = 0;

                    // ✅ CORREÇÃO #1: Criar um único escopo para a época e carregar os pesos uma vez.
                    using (var epochScope = new TensorScope($"Epoch_{epoch + 1}", _mathEngine,
                               currentModel.GetTensorManager()))
                    {
                        Console.WriteLine(
                            $"[Trainer] Carregando pesos do modelo para a VRAM para a Época {epoch + 1}...");
                        var weights = new ModelWeights
                        {
                            Embedding = epochScope.LoadTensor(currentModel.GetWeightsEmbeddingId()),
                            W_if = epochScope.LoadTensor(currentModel.GetWeightsInputForgetId()),
                            W_hf = epochScope.LoadTensor(currentModel.GetWeightsHiddenForgetId()),
                            B_f = epochScope.LoadTensor(currentModel.GetBiasForgetId()),
                            W_ii = epochScope.LoadTensor(currentModel.GetWeightsInputInputId()),
                            W_hi = epochScope.LoadTensor(currentModel.GetWeightsHiddenInputId()),
                            B_i = epochScope.LoadTensor(currentModel.GetBiasInputId()),
                            W_ic = epochScope.LoadTensor(currentModel.GetWeightsInputCellId()),
                            W_hc = epochScope.LoadTensor(currentModel.GetWeightsHiddenCellId()),
                            B_c = epochScope.LoadTensor(currentModel.GetBiasCellId()),
                            W_io = epochScope.LoadTensor(currentModel.GetWeightsInputOutputId()),
                            W_ho = epochScope.LoadTensor(currentModel.GetWeightsHiddenOutputId()),
                            B_o = epochScope.LoadTensor(currentModel.GetBiasOutputId()),
                            W_hy = epochScope.LoadTensor(currentModel.GetWeightsHiddenOutputFinalId()),
                            B_y = epochScope.LoadTensor(currentModel.GetBiasOutputFinalId()),
                            LN_f_gamma = epochScope.LoadTensor(currentModel.GetLnForgetGammaId()),
                            LN_f_beta = epochScope.LoadTensor(currentModel.GetLnForgetBetaId()),
                            LN_i_gamma = epochScope.LoadTensor(currentModel.GetLnInputGammaId()),
                            LN_i_beta = epochScope.LoadTensor(currentModel.GetLnInputBetaId()),
                            LN_c_gamma = epochScope.LoadTensor(currentModel.GetLnCellGammaId()),
                            LN_c_beta = epochScope.LoadTensor(currentModel.GetLnCellBetaId()),
                            LN_o_gamma = epochScope.LoadTensor(currentModel.GetLnOutputGammaId()),
                            LN_o_beta = epochScope.LoadTensor(currentModel.GetLnOutputBetaId())
                        };
                        Console.WriteLine("[Trainer] Pesos do modelo carregados na VRAM e prontos para a época.");

                        foreach (var batchIndex in trainBatchIndices)
                        {
                            _loteswatch.Start();
                            List<(int[] InputIndices, int[] TargetIndices)>? batch = null;
                            try
                            {
                                batch = datasetService.LoadBatchFromDisk(batchIndex);
                                if (batch == null || !batch.Any())
                                {
                                    Console.WriteLine($"[AVISO] Batch {batchIndex} vazio ou nulo, pulando...");
                                    continue;
                                }

                                double currentBatchLoss = 0;
                                foreach (var (inputIndices, targetIndices) in batch)
                                {
                                    if (inputIndices == null || targetIndices == null || !inputIndices.Any() ||
                                        !targetIndices.Any())
                                    {
                                        Console.WriteLine($"[AVISO] Sequência nula ou vazia no batch {batchIndex}, pulando...");
                                        continue;
                                    }

                                    if (currentModel != null)
                                    {
                                        // ✅ CORREÇÃO #2: Passar o objeto 'weights' pré-carregado para TrainSequence.
                                        currentBatchLoss += currentModel.TrainSequence(inputIndices, targetIndices,
                                            learningRate, weights);
                                    }
                                }

                                totalEpochLoss += currentBatchLoss;
                                batchCount++;
                                double avgBatchLoss = currentBatchLoss / batch.Count;
                                Console.WriteLine(
                                    $"Época: {epoch + 1}/{epochs} | Lotes: {batchCount}/{trainBatchIndices.Count}" +
                                    $" | Perda do Lote: {avgBatchLoss:F4}" +
                                    $"| Tempo do Lote : {_loteswatch.Elapsed:g}");
                                _loteswatch.Restart();

                                if (_mathEngine is GpuMathEngine gpuEngine2)
                                {
                                    gpuEngine2.Synchronize();
                                    gpuEngine2.FlushQueue();
                                }

                                if (batchCount % 10 == 0 && _gpuMemTracker != null)
                                {
                                    int orphans = _gpuMemTracker.CleanupOrphans();
                                    if (orphans > 0)
                                    {
                                        Console.WriteLine($"🧹 Removidos {orphans} tensores órfãos da VRAM");
                                    }

                                    _gpuMemTracker.PrintReport();
                                }

                                if (batchCount % 10 == 0)
                                {
                                    long currentMemoryMB = GetCurrentMemoryUsageMB();
                                    if (currentMemoryMB > MEMORY_TRIM_THRESHOLD_MB &&
                                        currentMemoryMB > (_lastTrimMemory + 1024))
                                    {
                                        Console.ForegroundColor = ConsoleColor.Red;
                                        Console.WriteLine(
                                            $"\n🔴 [CRITICAL RAM] Uso: {currentMemoryMB}MB > Limite: {CRITICAL_MEMORY_MB}MB. Executando limpeza de emergência...");
                                        Console.ResetColor();
                                        if (_syncGuard != null)
                                        {
                                            _syncGuard.SynchronizeBeforeRead("EmergencyCleanup");
                                        }

                                        ForceAggressiveGarbageCollection();
                                        Thread.Sleep(100);
                                        ForceAggressiveGarbageCollection();

                                        long memoryAfterGC = GetCurrentMemoryUsageMB();
                                        Console.ForegroundColor = ConsoleColor.Green;
                                        Console.WriteLine(
                                            $"✓ Memória após limpeza: {memoryAfterGC}MB (Liberado: {currentMemoryMB - memoryAfterGC}MB)");
                                        Console.ResetColor();
                                        _lastTrimMemory = memoryAfterGC;
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                failedBatches++;
                                Console.ForegroundColor = ConsoleColor.Red;
                                Console.WriteLine(
                                    $"[ERRO] Falha crítica no processamento do lote {batchIndex}: {ex.Message}\n{ex.StackTrace}");
                                Console.ResetColor();
                                if (failedBatches > 5)
                                    throw new Exception(
                                        $"Muitos lotes corrompidos ({failedBatches}). Abortando época.");
                                continue;
                            }
                            finally
                            {
                                if (batch != null)
                                {
                                    CleanupSingleBatch(batch);
                                    batch = null;
                                }
                            }
                        }
                    } // ✅ CORREÇÃO #3: O 'using (epochScope)' termina aqui, liberando os pesos da VRAM no final da época.

                    _stopwatch.Stop();
                    totalElapsedTime += _stopwatch.Elapsed;
                    double avgLoss =
                        batchCount > 0 ? totalEpochLoss / (batchCount * batchSize) : float.PositiveInfinity;
                    string elapsedFormatted =
                        $"{(int)_stopwatch.Elapsed.TotalHours:D2}:{_stopwatch.Elapsed.Minutes:D2}:{_stopwatch.Elapsed.Seconds:D2}";
                    File.AppendAllText(logPath,
                        $"Época {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4}. Duração: {elapsedFormatted}{Environment.NewLine}");
                    Console.WriteLine(
                        $"\nÉpoca {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4} | Duração: {elapsedFormatted}");

                    if (currentModel != null)
                    {
                        double validationLoss = ValidateModel(currentModel, datasetService, validationBatchIndices);
                        File.AppendAllText(logPath,
                            $"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}{Environment.NewLine}");
                        Console.WriteLine($"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}");
                        currentModel.ClearEpochTemporaryTensors();
                    }

                    long memoryBefore = GetCurrentMemoryUsageMB();
                    string modelPathForEpoch =
                        Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_{epoch + 1}.json");

                    Console.WriteLine($"\n╔════ LIBERAÇÃO COMPLETA DE MEMÓRIA (Fim da Época {epoch + 1}) ════╗");

                    if (currentModel != null)
                    {
                        Console.WriteLine($"[1/5] Salvando modelo em {modelPathForEpoch}...");
                        currentModel.SaveModel(modelPathForEpoch);
                        Console.WriteLine("[2/5] Zerando estados do AdamOptimizer...");
                        currentModel.ResetOptimizerState();
                        Console.WriteLine("[3/5] Descartando modelo atual da memória...");
                        currentModel.Dispose();
                        currentModel = null;
                    }

                    Console.WriteLine("[4/5] Forçando coleta de lixo em 3 estágios...");
                    ForceAggressiveGarbageCollection();

                    long memoryAfter = GetCurrentMemoryUsageMB();
                    long memoryFreed = memoryBefore - memoryAfter;
                    Console.ForegroundColor = memoryFreed > 0 ? ConsoleColor.Green : ConsoleColor.Yellow;
                    Console.WriteLine(
                        $"[Resultado] Memória ANTES: {memoryBefore}MB → DEPOIS: {memoryAfter}MB | Liberada: {memoryFreed}MB");
                    Console.ResetColor();

                    if (epoch < epochs - 1)
                    {
                        Console.WriteLine($"\n[5/5] Recarregando modelo para Época {epoch + 2}...");
                        var vocabManager = new VocabularyManager();
                        vocabManager.LoadVocabulary();
                        currentModel = GenerativeNeuralNetworkLSTM.Load(modelPathForEpoch, _mathEngine, vocabManager,
                            _searchService);
                        if (currentModel == null)
                            throw new InvalidOperationException(
                                $"CRÍTICO: Falha ao recarregar o modelo {modelPathForEpoch}.");

                        Console.WriteLine($"[Recarga] Memória atual: {GetCurrentMemoryUsageMB()}MB");
                        var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                        var estimatedTimeRemaining =
                            TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                        Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
                    }
                    else
                    {
                        Console.WriteLine("\n[5/5] Última época concluída. Não é necessário recarregar.");
                    }

                    Console.WriteLine("╚══════════════════════════════════════════════════════════╝\n");
                }

                if (currentModel == null && epochs > 0)
                {
                    Console.WriteLine("[Trainer] Recarregando modelo final para retorno...");
                    var vocabManager = new VocabularyManager();
                    vocabManager.LoadVocabulary();
                    string lastModelPath =
                        Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_{epochs}.json");
                    currentModel =
                        GenerativeNeuralNetworkLSTM.Load(lastModelPath, _mathEngine, vocabManager, _searchService);
                    if (currentModel == null)
                        throw new InvalidOperationException("CRÍTICO: Falha ao recarregar o modelo final.");
                }

                return currentModel;
            }
        }

        private void CleanupSingleBatch(List<(int[] InputIndices, int[] TargetIndices)> batch)
        {
            if (batch == null) return;
            batch.Clear();
            batch.TrimExcess();
        }

        private void ForceAggressiveGarbageCollection()
        {
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true, true);
        }

        private long GetCurrentMemoryUsageMB()
        {
            _currentProcess.Refresh();
            long currentMemory = _currentProcess.WorkingSet64 / (1024 * 1024);
            if (currentMemory > _peakMemoryUsageMB) _peakMemoryUsageMB = currentMemory;
            return currentMemory;
        }

        private double ValidateModel(GenerativeNeuralNetworkLSTM modelToValidate, DatasetService datasetService,
            List<long> validationBatchIndices)
        {
            Console.WriteLine("\n[Validação] Iniciando...");
            double totalLoss = 0;
            int sequenceCount = 0;
            var validationStopwatch = Stopwatch.StartNew();

            foreach (var batchIndex in validationBatchIndices)
            {
                var batch = datasetService.LoadBatchFromDisk(batchIndex);
                if (batch == null) continue;
                try
                {
                    foreach (var (inputIndices, targetIndices) in batch)
                    {
                        totalLoss += modelToValidate.CalculateSequenceLoss(inputIndices, targetIndices);
                        sequenceCount++;
                    }
                }
                finally
                {
                    CleanupSingleBatch(batch);
                }
            }

            validationStopwatch.Stop();
            Console.WriteLine(
                $"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss}. RAM: {GetCurrentMemoryUsageMB()}MB");

            return sequenceCount > 0 ? totalLoss / sequenceCount : double.PositiveInfinity;
        }
    }
}
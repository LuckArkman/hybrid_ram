using System.Diagnostics;
using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// 🔥 ZERO-RAM LSTM: TUDO em disco, NADA em RAM.
    /// Forward/Backward usam apenas swap files.
    /// </summary>
    public class NeuralNetworkLSTM : IDisposable
    {
        protected readonly AdamOptimizer _adamOptimizer;
        protected readonly IndividualFileTensorManager _tensorManager;
        protected readonly IMathEngine _mathEngine;

        // 🔥 NOVO: Swap manager substitui graph cache
        public readonly DiskSwapManager _swapManager;

        private readonly int inputSize;
        private readonly int hiddenSize;
        public readonly int outputSize;
        private readonly string _sessionId;
        private bool _disposed = false;
        public int warmupSteps;

        // IDs dos pesos (apenas ponteiros, não tensores)
        protected string _weightsEmbeddingId = null!;
        protected string _weightsInputForgetId = null!;
        protected string _weightsHiddenForgetId = null!;
        protected string _weightsInputInputId = null!;
        protected string _weightsHiddenInputId = null!;
        protected string _weightsInputCellId = null!;
        protected string _weightsHiddenCellId = null!;
        protected string _weightsInputOutputId = null!;
        protected string _weightsHiddenOutputId = null!;
        protected string _biasForgetId = null!;
        protected string _biasInputId = null!;
        protected string _biasCellId = null!;
        protected string _biasOutputId = null!;
        protected string _weightsHiddenOutputFinalId = null!;
        protected string _biasOutputFinalId = null!;
        protected string _hiddenStateId = null!;
        protected string _cellStateId = null!;
        protected string _lnForgetGammaId = null!;
        protected string _lnForgetBetaId = null!;
        protected string _lnInputGammaId = null!;
        protected string _lnInputBetaId = null!;
        protected string _lnCellGammaId = null!;
        protected string _lnCellBetaId = null!;
        protected string _lnOutputGammaId = null!;
        protected string _lnOutputBetaId = null!;

        public int InputSize => inputSize;
        public int HiddenSize => hiddenSize;
        public int OutputSize => outputSize;
        public IMathEngine GetMathEngine() => _mathEngine;

        public NeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, int outputSize,
            IMathEngine mathEngine)
        {
            this.inputSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this._mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            this._sessionId = $"session_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";

            // 🔥 SWAP MANAGER substitui graph cache
            this._swapManager = new DiskSwapManager(mathEngine, _sessionId);
            this._tensorManager = new IndividualFileTensorManager(mathEngine, _sessionId);
            this._adamOptimizer = new AdamOptimizer(_tensorManager);

            Console.WriteLine("╔═══════════════════════════════════════════════════════════╗");
            Console.WriteLine("║   🔥 ZERO-RAM LSTM (100% DISK-BACKED)                   ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");

            var rand = new Random(42);

            // Inicialização normal dos pesos
            _weightsEmbeddingId = InitializeWeight(vocabSize, embeddingSize, rand, "WeightsEmbedding");
            _weightsInputForgetId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputForget");
            _weightsHiddenForgetId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenForget");
            _weightsInputInputId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputInput");
            _weightsHiddenInputId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenInput");
            _weightsInputCellId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputCell");
            _weightsHiddenCellId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenCell");
            _weightsInputOutputId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputOutput");
            _weightsHiddenOutputId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenOutput");
            _biasForgetId = InitializeWeight(1, hiddenSize, rand, "BiasForget");
            _biasInputId = InitializeWeight(1, hiddenSize, rand, "BiasInput");
            _biasCellId = InitializeWeight(1, hiddenSize, rand, "BiasCell");
            _biasOutputId = InitializeWeight(1, hiddenSize, rand, "BiasOutput");
            _weightsHiddenOutputFinalId = InitializeWeight(hiddenSize, outputSize, rand, "WeightsOutputFinal");
            _biasOutputFinalId = InitializeWeight(1, outputSize, rand, "BiasOutputFinal");

            _hiddenStateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "HiddenState");
            _cellStateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "CellState");

            _lnForgetGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(),
                new[] { 1, hiddenSize }, "LN_Forget_Gamma");
            _lnForgetBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Forget_Beta");
            _lnInputGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(),
                new[] { 1, hiddenSize }, "LN_Input_Gamma");
            _lnInputBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Input_Beta");
            _lnCellGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(),
                new[] { 1, hiddenSize }, "LN_Cell_Gamma");
            _lnCellBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Cell_Beta");
            _lnOutputGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(),
                new[] { 1, hiddenSize }, "LN_Output_Gamma");
            _lnOutputBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Output_Beta");

            var stats = _tensorManager.GetStatistics();
            Console.WriteLine($"[ZERO-RAM LSTM] Inicialização completa:");
            Console.WriteLine($"  ├─ Tensores em disco: {stats.Count}");
            Console.WriteLine($"  ├─ Espaço em disco: {stats.DiskMB} MB");
            Console.WriteLine($"  └─ RAM usada (índices): {stats.RamMB} MB\n");
        }

        protected NeuralNetworkLSTM(NeuralNetworkLSTM existingModel)
        {
            this.inputSize = existingModel.inputSize;
            this.hiddenSize = existingModel.hiddenSize;
            this.outputSize = existingModel.outputSize;
            this._mathEngine = existingModel._mathEngine;
            this._adamOptimizer = existingModel._adamOptimizer;
            this._sessionId = existingModel._sessionId;
            this._tensorManager = existingModel._tensorManager;
            this._swapManager = existingModel._swapManager;

            // Copia IDs dos pesos
            this._weightsEmbeddingId = existingModel._weightsEmbeddingId;
            this._weightsInputForgetId = existingModel._weightsInputForgetId;
            this._weightsHiddenForgetId = existingModel._weightsHiddenForgetId;
            this._weightsInputInputId = existingModel._weightsInputInputId;
            this._weightsHiddenInputId = existingModel._weightsHiddenInputId;
            this._weightsInputCellId = existingModel._weightsInputCellId;
            this._weightsHiddenCellId = existingModel._weightsHiddenCellId;
            this._weightsInputOutputId = existingModel._weightsInputOutputId;
            this._weightsHiddenOutputId = existingModel._weightsHiddenOutputId;
            this._biasForgetId = existingModel._biasForgetId;
            this._biasInputId = existingModel._biasInputId;
            this._biasCellId = existingModel._biasCellId;
            this._biasOutputId = existingModel._biasOutputId;
            this._weightsHiddenOutputFinalId = existingModel._weightsHiddenOutputFinalId;
            this._biasOutputFinalId = existingModel._biasOutputFinalId;
            this._hiddenStateId = existingModel._hiddenStateId;
            this._cellStateId = existingModel._cellStateId;
        }

        private float[] CreateOrthogonalMatrix(int rows, int cols, Random rand)
        {
            var M = Matrix<float>.Build.Dense(rows, cols);
            var normalDist = new Normal(0, 1, rand);

            for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                M[i, j] = (float)normalDist.Sample();

            var svd = M.Svd(true);
            Matrix<float> orthogonalMatrix = rows >= cols ? svd.U : svd.VT.Transpose();

            if (orthogonalMatrix.RowCount != rows || orthogonalMatrix.ColumnCount != cols)
            {
                var finalMatrix = Matrix<float>.Build.Dense(rows, cols);
                finalMatrix.SetSubMatrix(0, 0,
                    orthogonalMatrix.SubMatrix(0, Math.Min(rows, orthogonalMatrix.RowCount), 0,
                        Math.Min(cols, orthogonalMatrix.ColumnCount)));
                return finalMatrix.ToColumnMajorArray();
            }

            return orthogonalMatrix.ToColumnMajorArray();
        }

        private string InitializeWeight(int rows, int cols, Random rand, string name)
        {
            float[] data;
            const float INITIALIZATION_VALUE_LIMIT = 0.5f;
            const int MAX_ATTEMPTS = 100;
            int attempt = 0;

            while (attempt < MAX_ATTEMPTS)
            {
                attempt++;

                if (name.Contains("BiasForget"))
                {
                    data = new float[rows * cols];
                    Array.Fill(data, 1.0f);
                }
                else if (name.Contains("WeightsHidden") && rows == cols)
                {
                    data = CreateOrthogonalMatrix(rows, cols, rand);
                }
                else
                {
                    data = new float[rows * cols];
                    double limit = Math.Sqrt(6.0 / ((double)rows + (double)cols));
                    for (int i = 0; i < data.Length; i++)
                        data[i] = (float)((rand.NextDouble() * 2 - 1) * limit);
                }

                bool isInitializationValid = true;
                if (!name.Contains("BiasForget"))
                {
                    foreach (float value in data)
                    {
                        if (float.IsNaN(value) || float.IsInfinity(value) ||
                            Math.Abs(value) > INITIALIZATION_VALUE_LIMIT)
                        {
                            isInitializationValid = false;
                            break;
                        }
                    }
                }

                if (isInitializationValid)
                {
                    if (attempt > 1)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"[InitializeWeight] ✓ Estabilizado '{name}' após {attempt} tentativas.");
                        Console.ResetColor();
                    }

                    return _tensorManager.CreateAndStore(data, new[] { rows, cols }, name);
                }
            }

            throw new InvalidOperationException(
                $"CRÍTICO: Falha ao inicializar '{name}' após {MAX_ATTEMPTS} tentativas.");
        }

        /// <summary>
        /// 🔥 FORWARD PASS 100% DISK-BASED (VERSÃO FINAL CORRIGIDA)
        /// </summary>
        protected (float totalLoss, List<string> swapFiles) ForwardPassZeroRAM(
            int[] inputIndices, int[] targetIndices, ModelWeights weights)
        {
            float sequenceLoss = 0;
            var swapFiles = new List<string>();
            string h_prev_swap, c_prev_swap;

            string initial_h_swap, initial_c_swap;

            using (var h_init = _tensorManager.LoadTensor(_hiddenStateId))
            {
                h_prev_swap = _swapManager.SwapOut(h_init, "h_init");
                initial_h_swap = h_prev_swap;
                swapFiles.Add(h_prev_swap);
            }

            using (var c_init = _tensorManager.LoadTensor(_cellStateId))
            {
                c_prev_swap = _swapManager.SwapOut(c_init, "c_init");
                initial_c_swap = c_prev_swap;
                swapFiles.Add(c_prev_swap);
            }

            for (int t = 0; t < inputIndices.Length; t++)
            {
                //Console.Write($"\r[Forward-ZeroRAM] Timestep {t + 1}/{inputIndices.Length}");

                string h_next_swap, c_next_swap;

                using (var h_prev = _swapManager.LoadFromSwap(h_prev_swap))
                using (var c_prev = _swapManager.LoadFromSwap(c_prev_swap))
                {
                    var inputEmbedding = _mathEngine.CreateTensor(new[] { 1, weights.Embedding.Shape[1] });
                    _mathEngine.Lookup(weights.Embedding, inputIndices[t], inputEmbedding);
                    swapFiles.Add(_swapManager.SwapOut(inputEmbedding, $"input_t{t}"));

                    // Forget Gate
                    var fg = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                    // ✅ CORREÇÃO: Removido o 9º argumento 'isForget'
                    ComputeGate(inputEmbedding, h_prev, weights.W_if, weights.W_hf, weights.B_f, weights.LN_f_gamma,
                        weights.LN_f_beta, fg);
                    var fg_swap = _swapManager.SwapOut(fg, $"fg_t{t}");
                    swapFiles.Add(fg_swap);

                    // Input Gate
                    var ig = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                    // ✅ CORREÇÃO: Removido o 9º argumento 'isForget'
                    ComputeGate(inputEmbedding, h_prev, weights.W_ii, weights.W_hi, weights.B_i, weights.LN_i_gamma,
                        weights.LN_i_beta, ig);
                    var ig_swap = _swapManager.SwapOut(ig, $"ig_t{t}");
                    swapFiles.Add(ig_swap);

                    var cc = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                    ComputeCellCandidate(inputEmbedding, h_prev, weights.W_ic, weights.W_hc, weights.B_c,
                        weights.LN_c_gamma, weights.LN_c_beta, cc);
                    var cc_swap = _swapManager.SwapOut(cc, $"cc_t{t}");
                    swapFiles.Add(cc_swap);

                    var c_next = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                    using (var fg_loaded = _swapManager.LoadFromSwap(fg_swap))
                    using (var ig_loaded = _swapManager.LoadFromSwap(ig_swap))
                    using (var cc_loaded = _swapManager.LoadFromSwap(cc_swap))
                    {
                        using var term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                        using var term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                        _mathEngine.Multiply(fg_loaded, c_prev, term1);
                        _mathEngine.Multiply(ig_loaded, cc_loaded, term2);
                        _mathEngine.Add(term1, term2, c_next);
                    }

                    c_next_swap = _swapManager.SwapOut(c_next, $"c_next_t{t}");
                    swapFiles.Add(c_next_swap);

                    // Output Gate
                    var og = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                    // ✅ CORREÇÃO: Removido o 9º argumento 'isForget'
                    ComputeGate(inputEmbedding, h_prev, weights.W_io, weights.W_ho, weights.B_o, weights.LN_o_gamma,
                        weights.LN_o_beta, og);
                    var og_swap = _swapManager.SwapOut(og, $"og_t{t}");
                    swapFiles.Add(og_swap);

                    var h_next = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                    using (var c_next_loaded = _swapManager.LoadFromSwap(c_next_swap))
                    {
                        var tanh_c = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                        _mathEngine.Tanh(c_next_loaded, tanh_c);
                        var tanh_c_swap = _swapManager.SwapOut(tanh_c, $"tanh_c_t{t}");
                        swapFiles.Add(tanh_c_swap);

                        using (var tanh_c_loaded = _swapManager.LoadFromSwap(tanh_c_swap))
                        using (var og_loaded = _swapManager.LoadFromSwap(og_swap))
                        {
                            _mathEngine.Multiply(og_loaded, tanh_c_loaded, h_next);
                        }
                    }

                    h_next_swap = _swapManager.SwapOut(h_next, $"h_next_t{t}");
                    swapFiles.Add(h_next_swap);

                    var pred = _mathEngine.CreateTensor(new[] { 1, outputSize });
                    using (var h_next_loaded = _swapManager.LoadFromSwap(h_next_swap))
                    {
                        using var outLinear = _mathEngine.CreateTensor(new[] { 1, outputSize });
                        _mathEngine.MatrixMultiply(h_next_loaded, weights.W_hy, outLinear);
                        _mathEngine.AddBroadcast(outLinear, weights.B_y, outLinear);
                        _mathEngine.Softmax(outLinear, pred);
                    }

                    using (var predCpu = pred.ToCpuTensor())
                    {
                        float prob = predCpu.GetData()[targetIndices[t]];
                        sequenceLoss += -MathF.Log(Math.Max(prob, 1e-9f));
                    }

                    swapFiles.Add(_swapManager.SwapOut(pred, $"pred_t{t}"));
                }

                h_prev_swap = h_next_swap;
                c_prev_swap = c_next_swap;
            }

            using (var final_h = _swapManager.LoadFromSwap(h_prev_swap))
            {
                _tensorManager.OverwriteTensor(_hiddenStateId, final_h);
            }

            using (var final_c = _swapManager.LoadFromSwap(c_prev_swap))
            {
                _tensorManager.OverwriteTensor(_cellStateId, final_c);
            }

            //Console.WriteLine($"\r[Forward-ZeroRAM] ✓ Completo. {swapFiles.Count} arquivos de swap criados.");
            return (sequenceLoss / inputIndices.Length, swapFiles);
        }

        private void ComputeGate(IMathTensor input, IMathTensor h_prev,
            IMathTensor W_i, IMathTensor W_h, IMathTensor bias,
            IMathTensor ln_gamma, IMathTensor ln_beta, IMathTensor result)
        {
            using var term1 = _mathEngine.CreateTensor(result.Shape);
            using var term2 = _mathEngine.CreateTensor(result.Shape);
            using var linear = _mathEngine.CreateTensor(result.Shape);

            _mathEngine.MatrixMultiply(input, W_i, term1);
            _mathEngine.MatrixMultiply(h_prev, W_h, term2);
            _mathEngine.Add(term1, term2, linear);
            _mathEngine.AddBroadcast(linear, bias, linear);
            _mathEngine.LayerNorm(linear, ln_gamma, ln_beta);
            _mathEngine.Sigmoid(linear, result);
        }

        private void ComputeCellCandidate(IMathTensor input, IMathTensor h_prev,
            IMathTensor W_i, IMathTensor W_h, IMathTensor bias,
            IMathTensor ln_gamma, IMathTensor ln_beta, IMathTensor result)
        {
            var term1 = _mathEngine.CreateTensor(result.Shape);
            var term2 = _mathEngine.CreateTensor(result.Shape);
            var linear = _mathEngine.CreateTensor(result.Shape);

            _mathEngine.MatrixMultiply(input, W_i, term1);
            _mathEngine.MatrixMultiply(h_prev, W_h, term2);
            _mathEngine.Add(term1, term2, linear);
            _mathEngine.AddBroadcast(linear, bias, linear);
            _mathEngine.LayerNorm(linear, ln_gamma, ln_beta);
            _mathEngine.Tanh(linear, result);

            term1.Dispose();
            term2.Dispose();
            linear.Dispose();
        }

        /// <summary>
        /// 🔥 TREINA UMA ÚNICA SEQUÊNCIA (VERSÃO CORRIGIDA E OTIMIZADA)
        /// Este método agora recebe os pesos do modelo pré-carregados para evitar I/O de disco
        /// e recriação de tensores a cada batch. Ele orquestra o ciclo completo de
        /// forward, backward e update, garantindo a limpeza de todos os tensores e arquivos temporários.
        /// </summary>
        /// <param name="inputIndices">Os índices dos tokens de entrada.</param>
        /// <param name="targetIndices">Os índices dos tokens de alvo (saída esperada).</param>
        /// <param name="learningRate">A taxa de aprendizado para esta sequência.</param>
        /// <param name="weights">O objeto contendo os tensores de peso do modelo, já carregados na VRAM.</param>
        /// <returns>A perda (loss) calculada para a sequência.</returns>
        public float TrainSequence(int[] inputIndices, int[] targetIndices, float learningRate, ModelWeights weights)
        {
            var sw = Stopwatch.StartNew();
            GpuSyncGuard? syncGuard = _mathEngine is GpuMathEngine gpuEngine ? gpuEngine.GetSyncGuard() : null;

            // O bloco 'finally' garante que a limpeza de swap files seja executada
            // mesmo que ocorra um erro durante o forward ou backward pass.
            try
            {
                float loss;
                List<string> forwardSwapFiles;
                Dictionary<string, string> gradIds;

                // NOTA DE CORREÇÃO: O 'using (var masterScope = ...)' e o carregamento dos pesos
                // foram removidos daqui e movidos para o ModelTrainerLSTM, para que ocorram
                // apenas uma vez por época, não a cada batch.

                // ==========================================================
                // FASE 1: FORWARD PASS
                // Executa a passagem para frente usando os pesos pré-carregados.
                // Retorna a perda e uma lista de arquivos de swap com os estados intermediários.
                // ==========================================================
                syncGuard?.SynchronizeBeforeRead("PreForward");
                (loss, forwardSwapFiles) = ForwardPassZeroRAM(inputIndices, targetIndices, weights);
                syncGuard?.SynchronizeBeforeRead("PostForward");

                // ==========================================================
                // FASE 2: BACKWARD PASS
                // Executa a retropropagação, lendo os swap files e calculando os gradientes.
                // Retorna um dicionário com os IDs dos arquivos de gradiente no disco.
                // ==========================================================
                gradIds = BackwardPassZeroRAM(inputIndices, targetIndices, forwardSwapFiles, weights);
                syncGuard?.SynchronizeBeforeRead("PostBackward");

                // ==========================================================
                // FASE 3: ATUALIZAÇÃO DOS PESOS (ADAM)
                // Mapeia os nomes dos pesos para seus IDs de tensor para o otimizador.
                // ==========================================================
                var weightIds = new Dictionary<string, string>
                {
                    { "W_embedding", _weightsEmbeddingId }, { "W_if", _weightsInputForgetId },
                    { "W_hf", _weightsHiddenForgetId }, { "B_f", _biasForgetId },
                    { "W_ii", _weightsInputInputId }, { "W_hi", _weightsHiddenInputId },
                    { "B_i", _biasInputId }, { "W_ic", _weightsInputCellId },
                    { "W_hc", _weightsHiddenCellId }, { "B_c", _biasCellId },
                    { "W_io", _weightsInputOutputId }, { "W_ho", _weightsHiddenOutputId },
                    { "B_o", _biasOutputId }, { "W_hy", _weightsHiddenOutputFinalId },
                    { "B_y", _biasOutputFinalId }, { "LN_f_gamma", _lnForgetGammaId },
                    { "LN_f_beta", _lnForgetBetaId }, { "LN_i_gamma", _lnInputGammaId },
                    { "LN_i_beta", _lnInputBetaId }, { "LN_c_gamma", _lnCellGammaId },
                    { "LN_c_beta", _lnCellBetaId }, { "LN_o_gamma", _lnOutputGammaId },
                    { "LN_o_beta", _lnOutputBetaId }
                };

                UpdateAdamGPUPassZeroRAM(weightIds, gradIds);
                syncGuard?.SynchronizeBeforeRead("PostAdamUpdate");

                // ==========================================================
                // FASE 4: LIMPEZA DE ARQUIVOS TEMPORÁRIOS
                // Deleta todos os arquivos temporários gerados neste ciclo.
                // ==========================================================
                foreach (var file in forwardSwapFiles)
                {
                    _swapManager.DeleteSwapFile(file);
                }

                foreach (var gradFileId in gradIds.Values)
                {
                    _tensorManager.DeleteTensor(gradFileId);
                }

                sw.Stop();
                // Console.WriteLine($"[TrainSequence-ZeroRAM] ⏱️ {sw.ElapsedMilliseconds}ms | Loss: {loss:F4}");
                return loss;
            }
            finally
            {
                _swapManager.ClearAllSwap();
                GC.Collect(0, GCCollectionMode.Forced, false);
            }
        }

        protected Dictionary<string, string> BackwardPassZeroRAM(int[] inputIndices,
            int[] targetIndices, List<string> forwardSwapFiles, ModelWeights weights)
        {
            var gradIds = new Dictionary<string, string>();

            // Etapa 1: Inicializar arquivos de gradiente em disco (zerados)
            gradIds["W_embedding"] = _tensorManager.CreateAndStoreZeros(weights.Embedding.Shape, "grad_Embedding");
            gradIds["W_if"] = _tensorManager.CreateAndStoreZeros(weights.W_if.Shape, "grad_W_if");
            gradIds["W_hf"] = _tensorManager.CreateAndStoreZeros(weights.W_hf.Shape, "grad_W_hf");
            gradIds["B_f"] = _tensorManager.CreateAndStoreZeros(weights.B_f.Shape, "grad_B_f");
            gradIds["W_ii"] = _tensorManager.CreateAndStoreZeros(weights.W_ii.Shape, "grad_W_ii");
            gradIds["W_hi"] = _tensorManager.CreateAndStoreZeros(weights.W_hi.Shape, "grad_W_hi");
            gradIds["B_i"] = _tensorManager.CreateAndStoreZeros(weights.B_i.Shape, "grad_B_i");
            gradIds["W_ic"] = _tensorManager.CreateAndStoreZeros(weights.W_ic.Shape, "grad_W_ic");
            gradIds["W_hc"] = _tensorManager.CreateAndStoreZeros(weights.W_hc.Shape, "grad_W_hc");
            gradIds["B_c"] = _tensorManager.CreateAndStoreZeros(weights.B_c.Shape, "grad_B_c");
            gradIds["W_io"] = _tensorManager.CreateAndStoreZeros(weights.W_io.Shape, "grad_W_io");
            gradIds["W_ho"] = _tensorManager.CreateAndStoreZeros(weights.W_ho.Shape, "grad_W_ho");
            gradIds["B_o"] = _tensorManager.CreateAndStoreZeros(weights.B_o.Shape, "grad_B_o");
            gradIds["W_hy"] = _tensorManager.CreateAndStoreZeros(weights.W_hy.Shape, "grad_W_hy");
            gradIds["B_y"] = _tensorManager.CreateAndStoreZeros(weights.B_y.Shape, "grad_B_y");
            gradIds["LN_f_gamma"] = _tensorManager.CreateAndStoreZeros(weights.LN_f_gamma.Shape, "grad_LN_f_gamma");
            gradIds["LN_f_beta"] = _tensorManager.CreateAndStoreZeros(weights.LN_f_beta.Shape, "grad_LN_f_beta");
            gradIds["LN_i_gamma"] = _tensorManager.CreateAndStoreZeros(weights.LN_i_gamma.Shape, "grad_LN_i_gamma");
            gradIds["LN_i_beta"] = _tensorManager.CreateAndStoreZeros(weights.LN_i_beta.Shape, "grad_LN_i_beta");
            gradIds["LN_c_gamma"] = _tensorManager.CreateAndStoreZeros(weights.LN_c_gamma.Shape, "grad_LN_c_gamma");
            gradIds["LN_c_beta"] = _tensorManager.CreateAndStoreZeros(weights.LN_c_beta.Shape, "grad_LN_c_beta");
            gradIds["LN_o_gamma"] = _tensorManager.CreateAndStoreZeros(weights.LN_o_gamma.Shape, "grad_LN_o_gamma");
            gradIds["LN_o_beta"] = _tensorManager.CreateAndStoreZeros(weights.LN_o_beta.Shape, "grad_LN_o_beta");

            // Etapa 2: Inicializar gradientes que se propagam para trás (dh e dc)
            string dh_next_swap, dc_next_swap;
            using (var scope = new TensorScope("BackwardInit", _mathEngine, _tensorManager))
            {
                var h_zeros = scope.CreateTensor(new[] { 1, hiddenSize });
                var c_zeros = scope.CreateTensor(new[] { 1, hiddenSize });
                dh_next_swap = _swapManager.SwapOut(h_zeros, "dh_next_init");
                dc_next_swap = _swapManager.SwapOut(c_zeros, "dc_next_init");
            }

            string FindSwapFile(string prefix) => forwardSwapFiles.First(f => Path.GetFileName(f).StartsWith(prefix));

            // Etapa 3: Loop de Backpropagation (de trás para frente)
            for (int t = targetIndices.Length - 1; t >= 0; t--)
            {
                //Console.Write($"\r[Backward-ZeroRAM] Timestep {t + 1}/{targetIndices.Length}");
                using (var scope = new TensorScope($"BPTT_t{t}", _mathEngine, _tensorManager))
                {
                    // Carrega todos os valores necessários do forward pass e os gradientes propagados
                    var pred = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"pred_t{t}")));
                    var h_next = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"h_next_t{t}")));
                    var tanh_c = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"tanh_c_t{t}")));
                    var og = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"og_t{t}")));
                    var cc = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"cc_t{t}")));
                    var ig = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"ig_t{t}")));
                    var fg = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"fg_t{t}")));
                    var c_prev =
                        scope.Track(
                            _swapManager.LoadFromSwap(t > 0
                                ? FindSwapFile($"c_next_t{t - 1}")
                                : FindSwapFile("c_init")));
                    var h_prev =
                        scope.Track(
                            _swapManager.LoadFromSwap(t > 0
                                ? FindSwapFile($"h_next_t{t - 1}")
                                : FindSwapFile("h_init")));
                    var input = scope.Track(_swapManager.LoadFromSwap(FindSwapFile($"input_t{t}")));
                    var dh_next = scope.Track(_swapManager.LoadFromSwap(dh_next_swap));
                    var dc_next = scope.Track(_swapManager.LoadFromSwap(dc_next_swap));

                    // --- Início dos Cálculos do Gradiente (LÓGICA CORRIGIDA) ---

                    // 1. Gradiente da saída (Softmax Cross-Entropy)
                    var d_pred = scope.Track(_mathEngine.Clone(pred));
                    using (var oneHot = _mathEngine.CreateOneHotTensor(new[] { targetIndices[t] }, outputSize))
                    {
                        _mathEngine.Subtract(d_pred, oneHot, d_pred);
                    }

                    // 2. Gradientes da camada de saída final (W_hy, B_y)
                    var d_Why = scope.Track(scope.CreateTensor(weights.W_hy.Shape));
                    _mathEngine.MatrixMultiplyTransposeA(h_next, d_pred, d_Why);
                    AccumulateGradientOnDisk(scope, gradIds["W_hy"], d_Why);
                    AccumulateGradientOnDisk(scope, gradIds["B_y"], d_pred);

                    // 3. Propagar gradiente para h_next (dh)
                    var dh = scope.Track(scope.CreateTensor(h_next.Shape));
                    _mathEngine.MatrixMultiplyTransposeB(d_pred, weights.W_hy, dh);
                    _mathEngine.Add(dh, dh_next, dh); // dh = d_output + dh_next

                    // 4. Gradiente para o estado da célula (dc)
                    var dc = scope.Track(
                        scope.CreateTensor(h_next
                            .Shape)); // ✅ CORREÇÃO: Usava h_next.Shape, semanticamente c_next.Shape é melhor
                    var d_tanh_c = scope.Track(scope.CreateTensor(tanh_c.Shape));
                    _mathEngine.Multiply(dh, og, d_tanh_c);
                    _mathEngine.TanhDerivative(tanh_c, tanh_c); // ✅ OTIMIZAÇÃO: Reuso de buffer
                    _mathEngine.Multiply(d_tanh_c, tanh_c, d_tanh_c);
                    _mathEngine.Add(dc_next, d_tanh_c, dc); // dc = dc_next + (dh * og * (1-tanh_c^2))

                    // 5. Gradiente da porta de saída (Output Gate)
                    var d_og = scope.Track(scope.CreateTensor(og.Shape));
                    _mathEngine.Multiply(dh, tanh_c, d_og);
                    _mathEngine.SigmoidDerivative(og, og); // ✅ OTIMIZAÇÃO: Reuso de buffer
                    _mathEngine.Multiply(d_og, og, d_og);

                    // 6. Gradiente do candidato a célula (Cell Candidate)
                    var d_cc = scope.Track(scope.CreateTensor(cc.Shape));
                    _mathEngine.Multiply(dc, ig, d_cc);
                    _mathEngine.TanhDerivative(cc, cc); // ✅ OTIMIZAÇÃO: Reuso de buffer
                    _mathEngine.Multiply(d_cc, cc, d_cc);

                    // 7. Gradiente da porta de entrada (Input Gate)
                    var d_ig = scope.Track(scope.CreateTensor(ig.Shape));
                    _mathEngine.Multiply(dc, cc, d_ig);
                    _mathEngine.SigmoidDerivative(ig, ig); // ✅ OTIMIZAÇÃO: Reuso de buffer
                    _mathEngine.Multiply(d_ig, ig, d_ig);

                    // 8. Gradiente da porta de esquecimento (Forget Gate)
                    var d_fg = scope.Track(scope.CreateTensor(fg.Shape));
                    _mathEngine.Multiply(dc, c_prev, d_fg);
                    _mathEngine.SigmoidDerivative(fg, fg); // ✅ OTIMIZAÇÃO: Reuso de buffer
                    _mathEngine.Multiply(d_fg, fg, d_fg);

                    // 9. Propagar gradientes para c_prev e h_prev (serão os 'next' para t-1)
                    var d_c_prev = scope.Track(scope.CreateTensor(c_prev.Shape));
                    _mathEngine.Multiply(dc, fg, d_c_prev);

                    var d_h_prev = scope.Track(scope.CreateTensor(h_prev.Shape)); // Acumulador
                    var d_input_acc = scope.Track(scope.CreateTensor(input.Shape)); // Acumulador

                    // Helper para retropropagar gradientes através de cada portão
                    Action<IMathTensor, IMathTensor, IMathTensor, string, string, string> backwardGate =
                        (d_gate, W_h, W_i, grad_Wh_id, grad_Wi_id, grad_B_id) =>
                        {
                            var d_Wh = scope.Track(scope.CreateTensor(W_h.Shape));
                            _mathEngine.MatrixMultiplyTransposeA(h_prev, d_gate, d_Wh);
                            AccumulateGradientOnDisk(scope, grad_Wh_id, d_Wh);

                            var d_Wi = scope.Track(scope.CreateTensor(W_i.Shape));
                            _mathEngine.MatrixMultiplyTransposeA(input, d_gate, d_Wi);
                            AccumulateGradientOnDisk(scope, grad_Wi_id, d_Wi);

                            AccumulateGradientOnDisk(scope, grad_B_id, d_gate);

                            var dh_prev_contrib = scope.Track(scope.CreateTensor(h_prev.Shape));
                            _mathEngine.MatrixMultiplyTransposeB(d_gate, W_h, dh_prev_contrib);
                            _mathEngine.Add(d_h_prev, dh_prev_contrib, d_h_prev);

                            var d_input_contrib = scope.Track(scope.CreateTensor(input.Shape));
                            _mathEngine.MatrixMultiplyTransposeB(d_gate, W_i, d_input_contrib);
                            _mathEngine.Add(d_input_acc, d_input_contrib, d_input_acc);
                        };

                    backwardGate(d_fg, weights.W_hf, weights.W_if, gradIds["W_hf"], gradIds["W_if"], gradIds["B_f"]);
                    backwardGate(d_ig, weights.W_hi, weights.W_ii, gradIds["W_hi"], gradIds["W_ii"], gradIds["B_i"]);
                    backwardGate(d_cc, weights.W_hc, weights.W_ic, gradIds["W_hc"], gradIds["W_ic"], gradIds["B_c"]);
                    backwardGate(d_og, weights.W_ho, weights.W_io, gradIds["W_ho"], gradIds["W_io"], gradIds["B_o"]);

                    // 10. Acumular gradiente do embedding
                    using (var d_embedding_matrix = scope.LoadTensor(gradIds["W_embedding"]))
                    {
                        _mathEngine.AccumulateGradient(d_embedding_matrix, d_input_acc,
                            inputIndices[t]); // ✅ CORREÇÃO: Usa inputIndices
                        _tensorManager.OverwriteTensor(gradIds["W_embedding"], d_embedding_matrix);
                    }

                    // 11. Atualizar os gradientes propagados para o próximo timestep (t-1)
                    _swapManager.DeleteSwapFile(dh_next_swap);
                    _swapManager.DeleteSwapFile(dc_next_swap);
                    dh_next_swap = _swapManager.SwapOut(d_h_prev, $"dh_next_t{t - 1}");
                    dc_next_swap = _swapManager.SwapOut(d_c_prev, $"dc_next_t{t - 1}");
                }
            }

            _swapManager.DeleteSwapFile(dh_next_swap);
            _swapManager.DeleteSwapFile(dc_next_swap);

            // Etapa 4: Pós-processamento dos gradientes acumulados
            ClipGradientsByValue(gradIds, 0.005f);
            ApplyGlobalGradientClipping(gradIds, maxNorm: 30.0f);

            //Console.WriteLine($"\r[Backward-ZeroRAM] ✓ Completo. {gradIds.Count} arquivos de gradiente atualizados.");
            return gradIds;
        }

        /// <summary>
        /// 🔥 ATUALIZA OS PESOS USANDO ADAM (VERSÃO CORRIGIDA E OTIMIZADA)
        /// Orquestra a atualização de todos os pesos do modelo usando o AdamOptimizer.
        /// Esta versão utiliza um único TensorScope para gerenciar todos os tensores
        /// (parâmetros, gradientes e estados do otimizador m/v) de forma eficiente,
        /// minimizando a sobrecarga e o risco de vazamentos de memória.
        /// </summary>
        /// <param name="weightIds">Dicionário mapeando nomes de pesos para seus IDs de tensor.</param>
        /// <param name="gradIds">Dicionário mapeando nomes de pesos para os IDs de seus gradientes correspondentes.</param>
        protected void UpdateAdamGPUPassZeroRAM(
            Dictionary<string, string> weightIds,
            Dictionary<string, string> gradIds)
        {
            // Console.WriteLine("\n[UpdatePass-ZeroRAM] Iniciando atualização de pesos via Adam...");
            var sw = Stopwatch.StartNew();

            // ✅ CORREÇÃO: Criar um único TensorScope que engloba TODO o processo de atualização.
            // Isso evita criar e destruir dezenas de scopes e tensores em um loop apertado.
            using (var updateScope = new TensorScope("AdamUpdate_FullPass", _mathEngine, _tensorManager))
            {
                int layerIndex = 0;
                foreach (var kvp in weightIds)
                {
                    string weightName = kvp.Key;

                    // Segurança: pular se não houver um gradiente correspondente para este peso.
                    if (!gradIds.TryGetValue(weightName, out var gradId))
                    {
                        // Console.WriteLine($"[UpdatePass-ZeroRAM] Aviso: Gradiente para '{weightName}' não encontrado. Pulando atualização.");
                        continue;
                    }

                    string weightId = kvp.Value;

                    // Carrega os tensores de parâmetro e gradiente DENTRO do escopo maior.
                    // Eles serão rastreados pelo 'updateScope'.
                    var paramTensor = updateScope.LoadTensor(weightId);
                    var gradTensor = updateScope.LoadTensor(gradId);

                    // O otimizador Adam também usa um TensorScope internamente para carregar
                    // seus estados 'm' e 'v', que serão liberados ao final de cada chamada.
                    _adamOptimizer.UpdateParametersGpu(layerIndex, paramTensor, gradTensor, _mathEngine);

                    // Salva o tensor de parâmetro (agora atualizado pelo Adam) de volta no disco.
                    _tensorManager.OverwriteTensor(weightId, paramTensor);

                    layerIndex++;

                    // Nenhum 'dispose' acontece aqui. Os tensores continuam vivos na VRAM,
                    // minimizando a sobrecarga de alocação/liberação a cada iteração.
                }
            } // ✅ TODOS os tensores (parâmetros e gradientes) carregados no loop são liberados AQUI, ao final de tudo.

            sw.Stop();
            // Console.WriteLine($"[UpdatePass-ZeroRAM] ✓ {weightIds.Count} tensores de peso atualizados em {sw.ElapsedMilliseconds}ms.");
        }

        /// <summary>
        /// 🔥 ACUMULAÇÃO DE GRADIENTE EM DISCO
        /// </summary>
        private void AccumulateGradientOnDisk(TensorScope scope, string accumulatedGradId, IMathTensor newGrad)
        {
            var accumulated_grad = scope.LoadTensor(accumulatedGradId);
            _mathEngine.Add(accumulated_grad, newGrad, accumulated_grad);
            _tensorManager.OverwriteTensor(accumulatedGradId, accumulated_grad);
        }

        // ==========================================================
        // MÉTODOS DE CLIPPING DE GRADIENTE (ZERO-RAM)
        // ==========================================================

        /// <summary>
        /// 🔥 SANEIA E LIMITA (CLIPS) GRADIENTES NO DISCO (ZERO-RAM)
        /// </summary>
        private void ClipGradientsByValue(Dictionary<string, string> gradientIds, float clipValue)
        {
            ///Console.WriteLine($"[ClipGradients] Saneando e limitando {gradientIds.Count} tensores de gradiente para o valor {clipValue}...");
            var sw = Stopwatch.StartNew();

            foreach (var gradId in gradientIds.Values)
            {
                using (var scope = new TensorScope($"ClipGrad_{Path.GetFileNameWithoutExtension(gradId)}", _mathEngine,
                           _tensorManager))
                {
                    var gradTensor = scope.LoadTensor(gradId);
                    _mathEngine.SanitizeAndClip(gradTensor, clipValue);
                    _tensorManager.OverwriteTensor(gradId, gradTensor);
                }
            }

            sw.Stop();
            //Console.WriteLine($"[ClipGradients] ✓ Sanitização e clipping concluídos em {sw.ElapsedMilliseconds}ms.");
        }

        /// <summary>
        /// 🔥 CALCULA A NORMA L2 TOTAL DOS GRADIENTES (ZERO-RAM)
        /// </summary>
        private float ComputeTotalNormZeroRAM(IEnumerable<string> gradientIds)
        {
            double totalSumOfSquares = 0.0;
            foreach (var gradId in gradientIds)
            {
                using (var scope = new TensorScope($"ComputeNorm_{Path.GetFileNameWithoutExtension(gradId)}",
                           _mathEngine, _tensorManager))
                {
                    var gradTensor = scope.LoadTensor(gradId);
                    totalSumOfSquares += _mathEngine.CalculateSumOfSquares(gradTensor);
                }
            }

            return MathF.Sqrt((float)totalSumOfSquares);
        }

        /// <summary>
        /// 🔥 APLICA GLOBAL GRADIENT CLIPPING (ZERO-RAM)
        /// </summary>
        private float ApplyGlobalGradientClipping(Dictionary<string, string> gradients, float maxNorm = 30.0f)
        {
            float totalNorm = ComputeTotalNormZeroRAM(gradients.Values);

            if (totalNorm <= maxNorm)
            {
                //Console.WriteLine($"[Clipping] Norma global {totalNorm:F4} ≤ {maxNorm:F1} → sem clipping necessário.");
                return totalNorm;
            }

            float scaleFactor = maxNorm / (totalNorm + 1e-8f);
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(
                $"[Clipping] NORMA GLOBAL {totalNorm:F4} > {maxNorm:F1} → aplicando fator de escala: {scaleFactor:F6}");
            Console.ResetColor();

            foreach (var gradId in gradients.Values)
            {
                using (var scope = new TensorScope($"ScaleGrad_{Path.GetFileNameWithoutExtension(gradId)}", _mathEngine,
                           _tensorManager))
                {
                    var gradTensor = scope.LoadTensor(gradId);
                    _mathEngine.Scale(gradTensor, scaleFactor);
                    _tensorManager.OverwriteTensor(gradId, gradTensor);
                }
            }

            float clippedNorm = totalNorm * scaleFactor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Clipping] Norma global após clipping: {clippedNorm:F4}");
            Console.ResetColor();

            return clippedNorm;
        }


        public void ResetHiddenState()
        {
            var zeros = new float[hiddenSize];
            _tensorManager.UpdateTensor(_hiddenStateId, t => t.UpdateFromCpu(zeros));
            _tensorManager.UpdateTensor(_cellStateId, t => t.UpdateFromCpu(zeros));
        }

        public void SaveModel(string filePath)
        {
            var embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];

            var modelData = new
            {
                VocabSize = inputSize,
                EmbeddingSize = embeddingSize,
                HiddenSize = hiddenSize,
                OutputSize = outputSize,
                SessionId = _sessionId,
                TensorIds = new Dictionary<string, string>
                {
                    ["WeightsEmbedding"] = _weightsEmbeddingId,
                    ["WeightsInputForget"] = _weightsInputForgetId,
                    ["WeightsHiddenForget"] = _weightsHiddenForgetId,
                    ["WeightsInputInput"] = _weightsInputInputId,
                    ["WeightsHiddenInput"] = _weightsHiddenInputId,
                    ["WeightsInputCell"] = _weightsInputCellId,
                    ["WeightsHiddenCell"] = _weightsHiddenCellId,
                    ["WeightsInputOutput"] = _weightsInputOutputId,
                    ["WeightsHiddenOutput"] = _weightsHiddenOutputId,
                    ["BiasForget"] = _biasForgetId,
                    ["BiasInput"] = _biasInputId,
                    ["BiasCell"] = _biasCellId,
                    ["BiasOutput"] = _biasOutputId,
                    ["WeightsOutputFinal"] = _weightsHiddenOutputFinalId,
                    ["BiasOutputFinal"] = _biasOutputFinalId,
                    ["HiddenState"] = _hiddenStateId,
                    ["CellState"] = _cellStateId
                }
            };

            string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, jsonString);
        }

        public static NeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
        {
            if (!File.Exists(filePath)) return null;

            string jsonString = File.ReadAllText(filePath);
            using var doc = JsonDocument.Parse(jsonString);
            var root = doc.RootElement;

            int vocabSize = root.GetProperty("VocabSize").GetInt32();
            int embeddingSize = root.GetProperty("EmbeddingSize").GetInt32();
            int hiddenSize = root.GetProperty("HiddenSize").GetInt32();
            int outputSize = root.GetProperty("OutputSize").GetInt32();

            var model = new NeuralNetworkLSTM(vocabSize, embeddingSize, hiddenSize, outputSize, mathEngine);

            var tensorIds = root.GetProperty("TensorIds");
            model._weightsEmbeddingId = tensorIds.GetProperty("WeightsEmbedding").GetString()!;
            model._weightsInputForgetId = tensorIds.GetProperty("WeightsInputForget").GetString()!;
            model._weightsHiddenForgetId = tensorIds.GetProperty("WeightsHiddenForget").GetString()!;
            model._weightsInputInputId = tensorIds.GetProperty("WeightsInputInput").GetString()!;
            model._weightsHiddenInputId = tensorIds.GetProperty("WeightsHiddenInput").GetString()!;
            model._weightsInputCellId = tensorIds.GetProperty("WeightsInputCell").GetString()!;
            model._weightsHiddenCellId = tensorIds.GetProperty("WeightsHiddenCell").GetString()!;
            model._weightsInputOutputId = tensorIds.GetProperty("WeightsInputOutput").GetString()!;
            model._weightsHiddenOutputId = tensorIds.GetProperty("WeightsHiddenOutput").GetString()!;
            model._biasForgetId = tensorIds.GetProperty("BiasForget").GetString()!;
            model._biasInputId = tensorIds.GetProperty("BiasInput").GetString()!;
            model._biasCellId = tensorIds.GetProperty("BiasCell").GetString()!;
            model._biasOutputId = tensorIds.GetProperty("BiasOutput").GetString()!;
            model._weightsHiddenOutputFinalId = tensorIds.GetProperty("WeightsOutputFinal").GetString()!;
            model._biasOutputFinalId = tensorIds.GetProperty("BiasOutputFinal").GetString()!;
            model._hiddenStateId = tensorIds.GetProperty("HiddenState").GetString()!;
            model._cellStateId = tensorIds.GetProperty("CellState").GetString()!;

            return model;
        }

        public void ResetOptimizerState()
        {
            _adamOptimizer.Reset();
        }

        public void ClearEpochTemporaryTensors()
        {
            // Limpa swap files
            _swapManager.ClearAllSwap();
        }

        public IndividualFileTensorManager GetTensorManager() => _tensorManager;
        public DiskSwapManager GetSwapManager() => _swapManager;

        public string GetWeightsEmbeddingId() => _weightsEmbeddingId;
        public string GetWeightsInputForgetId() => _weightsInputForgetId;
        public string GetWeightsHiddenForgetId() => _weightsHiddenForgetId;
        public string GetBiasForgetId() => _biasForgetId;
        public string GetWeightsInputInputId() => _weightsInputInputId;
        public string GetWeightsHiddenInputId() => _weightsHiddenInputId;
        public string GetBiasInputId() => _biasInputId;
        public string GetWeightsInputCellId() => _weightsInputCellId;
        public string GetWeightsHiddenCellId() => _weightsHiddenCellId;
        public string GetBiasCellId() => _biasCellId;
        public string GetWeightsInputOutputId() => _weightsInputOutputId;
        public string GetWeightsHiddenOutputId() => _weightsHiddenOutputId;
        public string GetBiasOutputId() => _biasOutputId;
        public string GetWeightsHiddenOutputFinalId() => _weightsHiddenOutputFinalId;
        public string GetBiasOutputFinalId() => _biasOutputFinalId;
        public string GetLnForgetGammaId() => _lnForgetGammaId;
        public string GetLnForgetBetaId() => _lnForgetBetaId;
        public string GetLnInputGammaId() => _lnInputGammaId;
        public string GetLnInputBetaId() => _lnInputBetaId;
        public string GetLnCellGammaId() => _lnCellGammaId;
        public string GetLnCellBetaId() => _lnCellBetaId;
        public string GetLnOutputGammaId() => _lnOutputGammaId;
        public string GetLnOutputBetaId() => _lnOutputBetaId;

        public (float, List<string>) RunForwardPassForInference(int[] inputIndices, int[] targetIndices,
            ModelWeights weights)
            => ForwardPassZeroRAM(inputIndices, targetIndices, weights);

        public void Dispose()
        {
            if (_disposed) return;

            Console.WriteLine("\n[Dispose] Finalizando ZERO-RAM LSTM...");
            _swapManager?.Dispose();
            _tensorManager?.Dispose();
            _adamOptimizer?.Reset();

            _disposed = true;
            GC.SuppressFinalize(this);
            Console.WriteLine("[Dispose] ✓ Limpeza completa.\n");
        }
    }
}
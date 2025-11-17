using System;
using System.IO;
using System.Collections.Generic;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// 🔥 ENHANCED SWAP MANAGER - Operações complexas 100% em disco
    /// Suporta operações binárias, ternárias e até quaternárias sem RAM.
    /// </summary>
    public class DiskSwapManager : IDisposable
    {
        private readonly string _swapDirectory;
        private readonly string _sessionId;
        private readonly IMathEngine _mathEngine;
        private readonly List<string> _swapFiles;
        private bool _disposed = false;
        
        // Estatísticas
        private long _totalSwapOps = 0;
        private long _totalBytesWritten = 0;
        private long _totalBytesRead = 0;

        public DiskSwapManager(IMathEngine mathEngine, string sessionId)
        {
            _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            _sessionId = sessionId;
            _swapDirectory = Path.Combine(Environment.CurrentDirectory, "Dayson", "Swap", sessionId);
            _swapFiles = new List<string>();

            if (Directory.Exists(_swapDirectory))
            {
                try { Directory.Delete(_swapDirectory, recursive: true); }
                catch (IOException) { }
            }

            Directory.CreateDirectory(_swapDirectory);
            Console.WriteLine($"[DiskSwap] 💾 Swap ativo em: {_swapDirectory}");
        }

        // Em DiskSwapManager.cs, substitua o método SwapOut existente por este:

        /// <summary>
        /// 🔥 Grava tensor em disco e DESCARTA da RAM imediatamente. (VERSÃO CORRIGIDA)
        /// Garante que o arquivo seja fisicamente escrito no disco antes de retornar.
        /// </summary>
        public string SwapOut(IMathTensor tensor, string label)
        {
            string swapFile = Path.Combine(_swapDirectory, $"{label}_{Guid.NewGuid():N}.swap");

            // Usa um FileStream com opções que forçam a escrita física e desabilitam o cache do SO.
            // FileOptions.WriteThrough é a chave para a correção.
            using (var fileStream = new FileStream(swapFile, FileMode.CreateNew, FileAccess.Write, FileShare.None, 
                       bufferSize: 4096, FileOptions.WriteThrough))
            using (var writer = new BinaryWriter(fileStream))
            {
                // Escreve metadados
                writer.Write(tensor.Shape.Length);
                foreach (var dim in tensor.Shape)
                    writer.Write(dim);
                writer.Write(tensor.Length);

                // Escreve dados
                tensor.WriteToStream(writer);

                // Força o flush de qualquer buffer intermediário do BinaryWriter
                writer.Flush();
        
                // Embora WriteThrough já faça isso, um flush explícito no FileStream
                // garante a intenção de forma inequívoca.
                fileStream.Flush(flushToDisk: true);
        
                _totalBytesWritten += fileStream.Position;
            }

            _swapFiles.Add(swapFile);
            _totalSwapOps++;

            // 🔥 CRÍTICO: Tensor é descartado IMEDIATAMENTE
            tensor.Dispose();

            return swapFile;
        }

        /// <summary>
        /// 🔥 Carrega tensor do disco temporariamente.
        /// DEVE ser usado dentro de using() para garantir dispose.
        /// </summary>
        public IMathTensor LoadFromSwap(string swapFile)
        {
            if (!File.Exists(swapFile))
                throw new FileNotFoundException($"Swap file não encontrado: {swapFile}");

            using (var fileStream = new FileStream(swapFile, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var reader = new BinaryReader(fileStream))
            {
                int shapeRank = reader.ReadInt32();
                int[] shape = new int[shapeRank];
                for (int i = 0; i < shapeRank; i++)
                    shape[i] = reader.ReadInt32();
                
                long length = reader.ReadInt64();
                var tensor = _mathEngine.CreateTensor(shape);
                tensor.ReadFromStream(reader, length);
                
                _totalBytesRead += fileStream.Position;
                return tensor;
            }
        }

        /// <summary>
        /// 🔥 Operação binária: (a OP b) → result_swap
        /// Ex: Add, Multiply, Subtract, etc.
        /// </summary>
        public string BinaryOp(string swapA, string swapB, 
            Action<IMathTensor, IMathTensor, IMathTensor> operation, string resultLabel)
        {
            using (var tensorA = LoadFromSwap(swapA))
            using (var tensorB = LoadFromSwap(swapB))
            using (var result = _mathEngine.CreateTensor(tensorA.Shape))
            {
                operation(tensorA, tensorB, result);
                return SwapOut(result, resultLabel);
            }
        }

        /// <summary>
        /// 🔥 Operação ternária: (a OP b OP c) → result_swap
        /// Ex: a*b + c
        /// </summary>
        public string TernaryOp(string swapA, string swapB, string swapC,
            Action<IMathTensor, IMathTensor, IMathTensor, IMathTensor> operation, string resultLabel)
        {
            using (var tensorA = LoadFromSwap(swapA))
            using (var tensorB = LoadFromSwap(swapB))
            using (var tensorC = LoadFromSwap(swapC))
            using (var result = _mathEngine.CreateTensor(tensorA.Shape))
            {
                // Executa operação customizada
                // Ex: _mathEngine.FusedMultiplyAdd(A, B, C, result)
                operation(tensorA, tensorB, tensorC, result);
                return SwapOut(result, resultLabel);
            }
        }

        /// <summary>
        /// 🔥 Operação unária com side-effect: (a OP) → a_modified_swap
        /// Ex: Sigmoid, Tanh, LayerNorm in-place
        /// </summary>
        public string UnaryInPlace(string swapA, 
            Action<IMathTensor, IMathTensor> operation, string resultLabel)
        {
            using (var tensorA = LoadFromSwap(swapA))
            using (var result = _mathEngine.CreateTensor(tensorA.Shape))
            {
                operation(tensorA, result);
                return SwapOut(result, resultLabel);
            }
        }

        /// <summary>
        /// 🔥 Matrix Multiply 100% disco: (A × B) → C_swap
        /// </summary>
        public string MatMul(string swapA, string swapB, int[] resultShape, string resultLabel)
        {
            using (var tensorA = LoadFromSwap(swapA))
            using (var tensorB = LoadFromSwap(swapB))
            using (var result = _mathEngine.CreateTensor(resultShape))
            {
                _mathEngine.MatrixMultiply(tensorA, tensorB, result);
                return SwapOut(result, resultLabel);
            }
        }

        /// <summary>
        /// 🔥 Operação complexa: Gate computation
        /// gate = sigmoid(input×W_i + hidden×W_h + bias)
        /// </summary>
        public string ComputeGate(string inputSwap, string hiddenSwap,
            string W_i_id, string W_h_id, string bias_id,
            string gamma_id, string beta_id, string resultLabel)
        {
            using (var input = LoadFromSwap(inputSwap))
            using (var hidden = LoadFromSwap(hiddenSwap))
            using (var W_i = _mathEngine.CreateTensor(new[] { input.Shape[1], hidden.Shape[1] })) // placeholder
            using (var W_h = _mathEngine.CreateTensor(new[] { hidden.Shape[1], hidden.Shape[1] }))
            using (var bias = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] }))
            using (var gamma = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] }))
            using (var beta = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] }))
            {
                // TODO: Carregar weights do TensorManager
                // Por enquanto placeholder
                
                var term1 = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] });
                var term2 = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] });
                var linear = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] });
                var result = _mathEngine.CreateTensor(new[] { 1, hidden.Shape[1] });

                try
                {
                    _mathEngine.MatrixMultiply(input, W_i, term1);
                    _mathEngine.MatrixMultiply(hidden, W_h, term2);
                    _mathEngine.Add(term1, term2, linear);
                    _mathEngine.AddBroadcast(linear, bias, linear);
                    _mathEngine.LayerNorm(linear, gamma, beta);
                    _mathEngine.Sigmoid(linear, result);

                    return SwapOut(result, resultLabel);
                }
                finally
                {
                    term1.Dispose();
                    term2.Dispose();
                    linear.Dispose();
                }
            }
        }

        /// <summary>
        /// 🔥 Element-wise multiply: (a * b) → result_swap
        /// </summary>
        public string Multiply(string swapA, string swapB, string resultLabel)
        {
            return BinaryOp(swapA, swapB, 
                (a, b, result) => _mathEngine.Multiply(a, b, result), 
                resultLabel);
        }

        /// <summary>
        /// 🔥 Element-wise add: (a + b) → result_swap
        /// </summary>
        public string Add(string swapA, string swapB, string resultLabel)
        {
            return BinaryOp(swapA, swapB,
                (a, b, result) => _mathEngine.Add(a, b, result),
                resultLabel);
        }

        /// <summary>
        /// 🔥 Tanh activation: tanh(a) → result_swap
        /// </summary>
        public string Tanh(string swapA, string resultLabel)
        {
            return UnaryInPlace(swapA,
                (input, result) => _mathEngine.Tanh(input, result),
                resultLabel);
        }

        /// <summary>
        /// 🔥 Sigmoid activation: sigmoid(a) → result_swap
        /// </summary>
        public string Sigmoid(string swapA, string resultLabel)
        {
            return UnaryInPlace(swapA,
                (input, result) => _mathEngine.Sigmoid(input, result),
                resultLabel);
        }

        /// <summary>
        /// 🔥 Deleta arquivo swap do disco.
        /// </summary>
        public void DeleteSwapFile(string swapFile)
        {
            if (string.IsNullOrEmpty(swapFile)) return;

            try
            {
                if (File.Exists(swapFile))
                {
                    //File.Delete(swapFile);
                    //_swapFiles.Remove(swapFile);
                }
            }
            catch (IOException ex)
            {
                Console.WriteLine($"[DiskSwap] ⚠️ Erro ao deletar: {ex.Message}");
            }
        }
        
        

        /// <summary>
        /// 🔥 Limpa TODOS os arquivos swap.
        /// </summary>
        public void ClearAllSwap()
        {
            int deleted = 0;
            foreach (var swapFile in _swapFiles.ToArray())
            {
                try
                {
                    if (File.Exists(swapFile))
                    {
                        File.Delete(swapFile);
                        deleted++;
                    }
                }
                catch { }
            }

            _swapFiles.Clear();
            
            if (deleted > 0)
            {
                //Console.WriteLine($"[DiskSwap] 🧹 Limpou {deleted} swap files");
            }

            // GC agressivo após limpeza
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        /// 📊 Imprime estatísticas de uso.
        /// </summary>
        public void PrintStats()
        {
            Console.WriteLine("\n╔═══════════════════════════════════════════════╗");
            Console.WriteLine("║       DISK SWAP MANAGER - ESTATÍSTICAS       ║");
            Console.WriteLine("╠═══════════════════════════════════════════════╣");
            Console.WriteLine($"║ Operações de swap:    {_totalSwapOps,10:N0}           ║");
            Console.WriteLine($"║ Arquivos ativos:      {_swapFiles.Count,10:N0}           ║");
            Console.WriteLine($"║ Bytes escritos:       {_totalBytesWritten / (1024.0 * 1024.0),10:F2} MB      ║");
            Console.WriteLine($"║ Bytes lidos:          {_totalBytesRead / (1024.0 * 1024.0),10:F2} MB      ║");
            Console.WriteLine($"║ Total I/O:            {(_totalBytesWritten + _totalBytesRead) / (1024.0 * 1024.0),10:F2} MB      ║");
            Console.WriteLine("╚═══════════════════════════════════════════════╝\n");
        }

        public void Dispose()
        {
            if (_disposed) return;

            PrintStats();
            ClearAllSwap();

            try
            {
                if (Directory.Exists(_swapDirectory))
                    Directory.Delete(_swapDirectory, recursive: true);
            }
            catch (IOException)
            {
                Console.WriteLine($"[DiskSwap] ⚠️ Não foi possível deletar diretório");
            }

            _disposed = true;
        }
    }
}
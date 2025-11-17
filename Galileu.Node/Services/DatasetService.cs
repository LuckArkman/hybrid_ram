using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Galileu.Node.TreeSwapFile; 

namespace Galileu.Node.Services
{
    public class DatasetService : IDisposable
    {
        private readonly string _swapFilePath;
        private readonly BinaryTreeFileStorage _batchStorage;
        private List<long> _trainBatchOffsets;
        private List<long> _validationBatchOffsets;
        private int _batchSize;
        private int _contextWindowSize;

        public DatasetService(string swapFilePath)
        {
            _swapFilePath = swapFilePath;
            
            var batchStoragePath = Path.Combine(Path.GetDirectoryName(swapFilePath) ?? "Dayson", "batches.bts"); 
            _batchStorage = new BinaryTreeFileStorage(batchStoragePath);

            _trainBatchOffsets = new List<long>(); 
            _validationBatchOffsets = new List<long>(); 
            _batchSize = 1;
            _contextWindowSize = 1;

            Console.WriteLine($"[DatasetService] Armazenamento de lotes BinaryTreeFileStorage inicializado em: {batchStoragePath}");
        }

        public void InitializeAndSplit(string text, int contextWindowSize, Dictionary<string, int> vocab, string padToken, int batchSize, float validationSplit)
        {
            Console.WriteLine($"[DatasetService] Iniciando processamento em streaming do dataset com contextWindowSize={contextWindowSize}, batchSize={batchSize}...");
            
            _batchStorage.Clear(); 
            _trainBatchOffsets.Clear();
            _validationBatchOffsets.Clear();
            _batchSize = batchSize;
            _contextWindowSize = contextWindowSize;
            
            // Etapa 1: Converter todo o texto em índices numéricos (única grande alocação temporária).
            var allTokens = text.Split(new[] {' '}, StringSplitOptions.RemoveEmptyEntries);
            var allIndices = allTokens.Select(t => vocab.ContainsKey(t) ? vocab[t] : vocab[padToken]).ToList();
            
            int totalSequences = Math.Max(0, allIndices.Count - contextWindowSize);
            int validationSize = (int)(totalSequences * validationSplit);
            int trainSize = totalSequences - validationSize;

            Console.WriteLine($"[DatasetService] Total de sequências a serem geradas: {totalSequences:N0}");
            Console.WriteLine($"[DatasetService] Treino: {trainSize:N0} sequências | Validação: {validationSize:N0} sequências");

            // Etapa 2: Processar lotes de treino em streaming, sem criar uma lista gigante de sequências.
            var batch = new List<(int[] InputIndices, int[] TargetIndices)>();
            for (int i = 0; i < trainSize; i++)
            {
                var sequence = GetSequenceAt(allIndices, i, contextWindowSize);
                batch.Add(sequence);

                if (batch.Count == batchSize)
                {
                    _trainBatchOffsets.Add(SaveBatchToDisk(batch));
                    batch.Clear(); // Libera a memória do lote imediatamente.
                }
            }
            if (batch.Any()) _trainBatchOffsets.Add(SaveBatchToDisk(batch)); // Salva o último lote parcial.
            batch.Clear();

            // Etapa 3: Processar lotes de validação em streaming.
            for (int i = trainSize; i < totalSequences; i++)
            {
                var sequence = GetSequenceAt(allIndices, i, contextWindowSize);
                batch.Add(sequence);

                if (batch.Count == batchSize)
                {
                    _validationBatchOffsets.Add(SaveBatchToDisk(batch));
                    batch.Clear();
                }
            }
            if (batch.Any()) _validationBatchOffsets.Add(SaveBatchToDisk(batch));
            batch.Clear();

            _batchStorage.Flush(); 
            
            // Etapa 4: Liberar a maior estrutura de dados da RAM e solicitar coleta de lixo.
            allIndices.Clear();
            allIndices.TrimExcess();
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();

            Console.WriteLine($"[DatasetService] Processamento em streaming concluído.");
            Console.WriteLine($"[DatasetService] Dados de treino: {_trainBatchOffsets.Count} lotes, Dados de validação: {_validationBatchOffsets.Count} lotes");
        }
        
        // Método auxiliar para criar uma única sequência sob demanda.
        private (int[] InputIndices, int[] TargetIndices) GetSequenceAt(List<int> allIndices, int startIndex, int contextWindowSize)
        {
            int sequenceLength = contextWindowSize - 1;
            var inputIndices = new int[sequenceLength];
            var targetIndices = new int[sequenceLength];

            for (int j = 0; j < sequenceLength; j++)
            {
                inputIndices[j] = allIndices[startIndex + j];
                targetIndices[j] = allIndices[startIndex + j + 1];
            }
            return (inputIndices, targetIndices);
        }

        private long SaveBatchToDisk(List<(int[] InputIndices, int[] TargetIndices)> batch)
        {
            try
            {
                byte[] dataBytes = SerializeBatch(batch);
                return _batchStorage.StoreData(dataBytes); 
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DatasetService] Erro ao salvar lote no BinaryTreeFileStorage: {ex.Message}");
                return -1;
            }
        }
        
        private byte[] SerializeBatch(List<(int[] InputIndices, int[] TargetIndices)> batch)
        {
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                writer.Write(batch.Count);
                foreach (var (inputIndices, targetIndices) in batch)
                {
                    writer.Write(inputIndices.Length);
                    writer.Write(targetIndices.Length);
                    foreach (var index in inputIndices) writer.Write(index);
                    foreach (var index in targetIndices) writer.Write(index);
                }
                return ms.ToArray();
            }
        }
        
        private List<(int[] InputIndices, int[] TargetIndices)> DeserializeBatch(byte[] data)
        {
            var batch = new List<(int[] InputIndices, int[] TargetIndices)>();
            using (var ms = new MemoryStream(data))
            using (var reader = new BinaryReader(ms))
            {
                int sequenceCount = reader.ReadInt32();
                for (int i = 0; i < sequenceCount; i++)
                {
                    int inputLength = reader.ReadInt32();
                    int targetLength = reader.ReadInt32();
                    
                    var inputIndices = new int[inputLength];
                    var targetIndices = new int[targetLength];
                    for (int j = 0; j < inputLength; j++) inputIndices[j] = reader.ReadInt32();
                    for (int j = 0; j < targetLength; j++) targetIndices[j] = reader.ReadInt32();

                    batch.Add((inputIndices, targetIndices));
                }
            }
            return batch;
        }

        // CORREÇÃO COMPLETA - DatasetService.cs
// Substitua o método LoadBatchFromDisk existente por este

        public List<(int[] InputIndices, int[] TargetIndices)>? LoadBatchFromDisk(long offset)
        {
            if (offset < 0) 
                throw new ArgumentException($"Offset inválido: {offset}");
    
            byte[] dataBytes = null;
    
            try
            {
                // ✅ Carrega dados binários do storage
                dataBytes = _batchStorage.GetDataBytes(offset);
        
                if (dataBytes == null || dataBytes.Length == 0)
                {
                    throw new InvalidDataException($"Dados vazios no offset {offset}");
                }
        
                // ✅ Deserializa batch
                var batch = DeserializeBatch(dataBytes);
        
                // ✅ Limpa array temporário IMEDIATAMENTE
                Array.Clear(dataBytes, 0, dataBytes.Length);
                dataBytes = null;
        
                return batch;
            }
            catch (InvalidDataException dataEx)
            {
                Console.WriteLine($"[DatasetService] Dados corrompidos no offset {offset}: {dataEx.Message}");
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DatasetService] Falha ao carregar batch no offset {offset}: {ex.Message}");
                throw;
            }
            finally
            {
                // ✅ CORREÇÃO: Garante cleanup do byte array
                if (dataBytes != null)
                {
                    try
                    {
                        Array.Clear(dataBytes, 0, dataBytes.Length);
                        dataBytes = null;
                
                        // Hint ao GC se array era grande (>10MB)
                        // GC.Collect(0, GCCollectionMode.Optimized, blocking: false);
                    }
                    catch
                    {
                        // Best effort - falha silenciosa no cleanup
                    }
                }
            }
        }

        public List<long> GetTrainBatchOffsets() => _trainBatchOffsets;

        public List<long> GetValidationBatchOffsets() => _validationBatchOffsets;
        
        public void Dispose()
        {
            long memoryBefore = GC.GetTotalMemory(false) / (1024 * 1024);
            _trainBatchOffsets.Clear();
            _trainBatchOffsets.TrimExcess();
            _validationBatchOffsets.Clear();
            _validationBatchOffsets.TrimExcess();
            
            _batchStorage?.Dispose(); 

            long memoryAfter = GC.GetTotalMemory(true) / (1024 * 1024);
            Console.WriteLine($"[DatasetService] Dispose concluído. Memória liberada: aprox. {memoryBefore - memoryAfter} MB");
        }
    }
}
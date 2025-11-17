using System.Collections.Concurrent;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Gerenciador de tensores que serializa cada tensor em seu próprio arquivo .bin individual.
    /// Esta abordagem é mais robusta para escritas concorrentes de alta frequência,
    /// pois cada operação de arquivo é atômica e isolada.
    /// </summary>
    public class IndividualFileTensorManager : IDisposable
    {
        private readonly IMathEngine _mathEngine;
        private readonly string _tensorDirectory;
        private readonly ConcurrentDictionary<string, int[]> _tensorIndex; // Apenas ID e Shape
        private readonly ConcurrentDictionary<string, object> _tensorUpdateLocks;
        private int _nextTensorId = 0;
        private bool _disposed = false;

        public IndividualFileTensorManager(IMathEngine mathEngine, string sessionId)
        {
            _mathEngine = mathEngine;
            _tensorDirectory = Path.Combine(Environment.CurrentDirectory, "Dayson", "TensorCache", sessionId);
            _tensorIndex = new ConcurrentDictionary<string, int[]>();
            _tensorUpdateLocks = new ConcurrentDictionary<string, object>();

            if (Directory.Exists(_tensorDirectory))
            {
                try
                {
                    Directory.Delete(_tensorDirectory, recursive: true);
                }
                catch (IOException)
                {
                    /* Ignora se houver lock */
                }
            }

            Directory.CreateDirectory(_tensorDirectory);

            Console.WriteLine(
                $"[IndividualFileTensorManager] Sessão '{sessionId}' inicializada em: {_tensorDirectory}");
        }

        private string GetPathForId(string id) => Path.Combine(_tensorDirectory, $"{id}.bin");

        /// <summary>
        /// Serializa um tensor em memória para um novo arquivo binário individual no disco
        /// e registra seus metadados (ID e shape) no índice em memória.
        /// </summary>
        /// <param name="tensor">O tensor em memória (CPU ou GPU) a ser armazenado.</param>
        /// <param name="name">Um nome descritivo para o tensor (ex: "ForgetGate_t15").</param>
        /// <returns>Um ID único e globalmente identificável para o tensor armazenado.</returns>
        /// <exception cref="ArgumentNullException">Lançada se o tensor ou o nome forem nulos.</exception>
        /// <exception cref="InvalidOperationException">Lançada se houver uma colisão de ID ao registrar no índice.</exception>
        /// <exception cref="IOException">Lançada em caso de falha de I/O durante a escrita do arquivo.</exception>
        public string StoreTensor(IMathTensor tensor, string name)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (string.IsNullOrEmpty(name))
                throw new ArgumentNullException(nameof(name));

            // Passo 1: Gerar um ID único e seguro para o arquivo.
            // Combina um nome descritivo, um contador sequencial para ordem e um GUID para unicidade.
            int uniqueSequenceId = Interlocked.Increment(ref _nextTensorId);
            string id = $"{name.Replace(" ", "_")}_{uniqueSequenceId:D8}_{Guid.NewGuid():N}";
            string filePath = GetPathForId(id);

            try
            {
                // Passo 2: Escrever o tensor no disco.
                // Usamos um FileStream com FileMode.CreateNew para garantir que a operação falhe
                // se, por alguma razão improvável, o arquivo já existir (prevenindo corrupção de dados).
                using (var fileStream = new FileStream(filePath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                using (var writer = new BinaryWriter(fileStream))
                {
                    // Formato de arquivo explícito para desserialização robusta:
                    // [int32: Rank do Shape] [int32...: Dimensões do Shape] [int64: Comprimento total] [float...: Dados]

                    // Escreve o cabeçalho de metadados.
                    writer.Write(tensor.Shape.Length);
                    foreach (var dim in tensor.Shape)
                    {
                        writer.Write(dim);
                    }

                    writer.Write(tensor.Length);

                    // Delega a escrita dos dados brutos para a implementação otimizada do tensor.
                    tensor.WriteToStream(writer);

                    // Garante que todos os buffers sejam escritos no disco físico.
                    writer.Flush();
                }

                // Passo 3: Registrar o novo tensor no índice em memória.
                // Esta é a etapa final. Se a escrita no disco falhou, esta linha não é alcançada.
                if (!_tensorIndex.TryAdd(id, (int[])tensor.Shape.Clone()))
                {
                    // Se o registro falhar (extremamente raro), limpamos o arquivo órfão
                    // que foi criado no disco para manter o sistema consistente.
                    try
                    {
                        File.Delete(filePath);
                    }
                    catch
                    {
                        /* melhor esforço para limpar */
                    }

                    throw new InvalidOperationException(
                        $"Falha crítica ao registrar o tensor com ID '{id}' no índice. Conflito de chave detectado.");
                }

                return id;
            }
            catch (IOException ex)
            {
                // Fornece um contexto útil para a exceção de I/O.
                throw new IOException(
                    $"Falha ao escrever o arquivo do tensor em '{filePath}'. Verifique as permissões, o caminho e o espaço em disco.",
                    ex);
            }
        }

        /// <summary>
        /// Desserializa um tensor do seu arquivo binário individual no disco para a memória (RAM ou VRAM).
        /// </summary>
        /// <param name="id">O ID único do tensor a ser carregado.</param>
        /// <returns>Uma nova instância de IMathTensor (CpuTensor ou GpuTensor) contendo os dados do disco.</returns>
        /// <exception cref="KeyNotFoundException">Lançada se o ID do tensor não for encontrado no índice em memória.</exception>
        /// <exception cref="FileNotFoundException">Lançada se o arquivo do tensor não for encontrado no disco, indicando um cache corrompido.</exception>
        /// <exception cref="InvalidDataException">Lançada se o conteúdo do arquivo estiver corrompido ou inconsistente (ex: shape não corresponde ao length).</exception>
        public IMathTensor LoadTensor(string id)
        {
            // Passo 1: Validação de Sanidade do ID e do Índice
            if (string.IsNullOrEmpty(id))
            {
                throw new ArgumentNullException(nameof(id), "O ID do tensor para carregar não pode ser nulo ou vazio.");
            }

            if (!_tensorIndex.TryGetValue(id, out var shapeFromIndex))
            {
                throw new KeyNotFoundException(
                    $"O ID de tensor '{id}' não foi encontrado no índice em memória. O tensor pode ter sido deletado ou nunca foi armazenado.");
            }

            string filePath = GetPathForId(id);

            // Passo 2: Validação da Existência do Arquivo Físico
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException(
                    $"Corrupção de cache detectada: O arquivo do tensor para o ID '{id}' não foi encontrado em '{filePath}', embora o ID exista no índice.",
                    filePath);
            }

            try
            {
                using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (var reader = new BinaryReader(fileStream))
                {
                    // Passo 3: Leitura e Validação do Cabeçalho de Metadados do Arquivo
                    if (fileStream.Length < sizeof(int) * 2 + sizeof(long)) // Validação mínima do tamanho do cabeçalho
                    {
                        throw new InvalidDataException(
                            $"Arquivo do tensor '{filePath}' está truncado ou corrompido. Tamanho insuficiente para o cabeçalho.");
                    }

                    int shapeRank = reader.ReadInt32();
                    if (shapeRank < 0 || shapeRank > 10) // Limite de sanidade para o número de dimensões
                    {
                        throw new InvalidDataException(
                            $"Shape rank inválido ({shapeRank}) lido do arquivo '{filePath}'.");
                    }

                    var shapeFromFile = new int[shapeRank];
                    for (int i = 0; i < shapeRank; i++)
                    {
                        shapeFromFile[i] = reader.ReadInt32();
                    }

                    long lengthFromFile = reader.ReadInt64();
                    long expectedLengthFromShape = shapeFromFile.Aggregate(1L, (a, b) => a * b);

                    if (lengthFromFile != expectedLengthFromShape)
                    {
                        throw new InvalidDataException(
                            $"Inconsistência de metadados no arquivo '{filePath}'. O shape [{string.Join("x", shapeFromFile)}] resulta em um comprimento de {expectedLengthFromShape}, mas o comprimento registrado no arquivo é {lengthFromFile}.");
                    }

                    // Passo 4: Criação do Tensor e Leitura dos Dados
                    // A MathEngine criará o tipo correto de tensor (CPU ou GPU).
                    // A responsabilidade de chamar .Dispose() neste tensor é do código que chamou LoadTensor
                    // (idealmente, através de um TensorScope).
                    var tensor = _mathEngine.CreateTensor(shapeFromFile);

                    // Delega a leitura para a implementação otimizada do tensor, que usa ArrayPool para
                    // evitar alocações de RAM para buffers intermediários.
                    tensor.ReadFromStream(reader, lengthFromFile);

                    return tensor;
                }
            }
            catch (EndOfStreamException ex)
            {
                throw new InvalidDataException(
                    $"Fim inesperado do arquivo ao ler o tensor '{filePath}'. O arquivo pode estar incompleto ou corrompido.",
                    ex);
            }
            catch (IOException ex)
            {
                throw new IOException($"Falha de I/O ao ler o arquivo do tensor '{filePath}'.", ex);
            }
        }
        
        /// <summary>
        /// Atualiza uma única linha (row) de um tensor 2D no disco com os dados de outro tensor.
        /// Esta é uma operação de Leitura-Modificação-Escrita.
        /// </summary>
        /// <param name="id">O ID do tensor no disco a ser modificado.</param>
        /// <param name="rowIndex">O índice da linha a ser atualizada.</param>
        /// <param name="sourceRowTensor">O tensor em memória contendo os novos dados para a linha.</param>
        public void SetTensorRow(string id, int rowIndex, IMathTensor sourceRowTensor)
        {
            var tensorLock = _tensorUpdateLocks.GetOrAdd(id, _ => new object());
            lock (tensorLock)
            {
                // Carrega o tensor completo do disco.
                using IMathTensor fullTensor = LoadTensor(id);

                if (fullTensor.Shape.Length != 2 || sourceRowTensor.Shape.Length != 2)
                    throw new InvalidOperationException("SetTensorRow só suporta tensores 2D.");
                if (fullTensor.Shape[1] != sourceRowTensor.Shape[1])
                    throw new ArgumentException("A largura (número de colunas) do tensor fonte não corresponde à do tensor de destino.");

                // Atualiza a linha específica na memória (CPU ou GPU).
                _mathEngine.Set(fullTensor, rowIndex, sourceRowTensor);

                // Salva o tensor modificado de volta para o disco, sobrescrevendo o arquivo antigo.
                OverwriteTensor(id, fullTensor);
            }
        }

        public bool TensorExists(string id)
        {
            if (string.IsNullOrEmpty(id)) return false;

            // A verificação mais confiável é a existência do arquivo físico.
            return File.Exists(GetPathForId(id));
        }

        public void DeleteTensor(string id)
        {
            if (string.IsNullOrEmpty(id)) return;

            _tensorIndex.TryRemove(id, out _);
            _tensorUpdateLocks.TryRemove(id, out _);
            try
            {
                string filePath = GetPathForId(id);
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                }
            }
            catch (IOException)
            {
                /* Ignora se o arquivo estiver bloqueado */
            }
        }

        public string CloneTensor(string sourceId, string newName)
        {
            string sourcePath = GetPathForId(sourceId);
            if (!_tensorIndex.TryGetValue(sourceId, out var shape) || !File.Exists(sourcePath))
            {
                throw new KeyNotFoundException($"Tensor fonte com ID '{sourceId}' não encontrado para clonagem.");
            }

            int uniqueSequenceId = Interlocked.Increment(ref _nextTensorId);
            string newId = $"{newName}_{uniqueSequenceId:D6}_{Guid.NewGuid():N}";
            string destPath = GetPathForId(newId);

            try
            {
                File.Copy(sourcePath, destPath, true);
                if (!_tensorIndex.TryAdd(newId, (int[])shape.Clone()))
                {
                    try
                    {
                        File.Delete(destPath);
                    }
                    catch
                    {
                        /* best effort */
                    }

                    throw new InvalidOperationException($"Falha ao registrar clone de tensor com ID '{newId}'.");
                }

                return newId;
            }
            catch (IOException ex)
            {
                throw new IOException($"Falha ao clonar arquivo de tensor de '{sourceId}' para '{newId}'.", ex);
            }
        }

        public string CreateAndStore(float[] data, int[] shape, string name)
        {
            using var tensor = _mathEngine.CreateTensor(data, shape);
            return StoreTensor(tensor, name);
        }

        public string CreateAndStoreZeros(int[] shape, string name)
        {
            using var tensor = _mathEngine.CreateTensor(shape);
            return StoreTensor(tensor, name);
        }

        public int[] GetShape(string id)
        {
            if (_tensorIndex.TryGetValue(id, out var shape))
            {
                return (int[])shape.Clone();
            }

            throw new KeyNotFoundException($"Tensor ID '{id}' não encontrado no índice.");
        }

        public void UpdateTensor(string id, Action<IMathTensor> operation)
        {
            var tensorLock = _tensorUpdateLocks.GetOrAdd(id, _ => new object());
            lock (tensorLock)
            {
                using IMathTensor tensor = LoadTensor(id);
                operation(tensor);

                string filePath = GetPathForId(id);
                using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
                using var writer = new BinaryWriter(fileStream);

                // Re-escreve no formato correto
                writer.Write(tensor.Shape.Length);
                foreach (var dim in tensor.Shape)
                {
                    writer.Write(dim);
                }

                writer.Write(tensor.Length);
                tensor.WriteToStream(writer);
            }
        }

        public List<string> ListTensors(string? nameFilter = null)
        {
            var query = _tensorIndex.Keys.AsEnumerable();
            if (!string.IsNullOrEmpty(nameFilter))
            {
                query = query.Where(id => id.StartsWith(nameFilter, StringComparison.OrdinalIgnoreCase));
            }

            return query.ToList();
        }

        public (int Count, long DiskMB, long RamMB, int TotalAccesses) GetStatistics()
        {
            long totalBytes = 0;
            try
            {
                if (Directory.Exists(_tensorDirectory))
                {
                    totalBytes = new DirectoryInfo(_tensorDirectory).GetFiles("*.bin").Sum(f => f.Length);
                }
            }
            catch
            {
            }

            long ramUsage = (_tensorIndex.Count * 128) + (_tensorUpdateLocks.Count * 64);
            return (_tensorIndex.Count, totalBytes / (1024 * 1024), ramUsage / (1024 * 1024), -1);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _tensorIndex.Clear();
            _tensorUpdateLocks.Clear();
            try
            {
                if (Directory.Exists(_tensorDirectory))
                {
                    Directory.Delete(_tensorDirectory, recursive: true);
                }
            }
            catch (IOException)
            {
            }

            _disposed = true;
        }

        /// <summary>
        /// Sobrescreve completamente o conteúdo de um tensor no disco com os dados de um tensor em memória.
        /// Esta operação é mais rápida que 'UpdateTensor' pois não realiza uma leitura prévia.
        /// É ideal para inicializar ou resetar o valor de um tensor em disco.
        /// </summary>
        /// <param name="id">O ID do tensor no disco a ser sobrescrito.</param>
        /// <param name="sourceTensor">O tensor em memória (CPU ou GPU) cujos dados serão escritos.</param>
        public void OverwriteTensor(string id, IMathTensor sourceTensor)
        {
            if (string.IsNullOrEmpty(id))
                throw new ArgumentException("O ID do tensor não pode ser nulo ou vazio.", nameof(id));
            if (sourceTensor == null)
                throw new ArgumentNullException(nameof(sourceTensor));
            if (!_tensorIndex.ContainsKey(id))
                throw new KeyNotFoundException(
                    $"Tensor ID '{id}' não encontrado no índice. Não é possível sobrescrever um tensor que não foi registrado.");

            // Adquire um lock específico para este ID de tensor para garantir que a escrita seja atômica
            // e não entre em conflito com outras leituras ou escritas.
            var tensorLock = _tensorUpdateLocks.GetOrAdd(id, _ => new object());

            lock (tensorLock)
            {
                string filePath = GetPathForId(id);

                try
                {
                    // Usa FileMode.Create para truncar o arquivo se ele existir ou criar um novo.
                    // Isso garante que estamos começando com um arquivo limpo para a sobrescrita.
                    using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
                    using var writer = new BinaryWriter(fileStream);

                    // Escreve o cabeçalho completo: Rank do Shape, Dimensões, Comprimento total
                    writer.Write(sourceTensor.Shape.Length); // int
                    foreach (var dim in sourceTensor.Shape)
                    {
                        writer.Write(dim); // int
                    }

                    writer.Write(sourceTensor.Length); // long

                    // Delega a escrita dos dados brutos para a implementação específica do tensor (CPU ou GPU).
                    // A implementação otimizada usará pooling de buffer para evitar alocações de RAM.
                    sourceTensor.WriteToStream(writer);

                    // Garante que todos os dados sejam gravados no disco físico.
                    writer.Flush();
                    fileStream.Flush(true);
                }
                catch (IOException ex)
                {
                    // Fornece um contexto mais útil para a exceção.
                    throw new IOException(
                        $"Falha de I/O ao tentar sobrescrever o arquivo do tensor '{filePath}'. Verifique as permissões e o espaço em disco.",
                        ex);
                }

                // Atualiza o shape no índice em memória, caso tenha mudado.
                _tensorIndex[id] = (int[])sourceTensor.Shape.Clone();
            }
        }
    }
}
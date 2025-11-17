using System;
using System.Collections.Generic;
using System.IO;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Gerencia o grafo computacional como um arquivo de cache no disco, evitando
    /// o acúmulo de nós do grafo na memória RAM durante o forward pass.
    /// Os nós são lidos de forma reversa e sob demanda (streaming) durante o backward pass.
    /// </summary>
    public class ComputationGraphCacheManager : IDisposable
    {
        private readonly string _cacheFilePath;
        private FileStream? _fileStream;
        private BinaryWriter? _writer;
        private readonly List<long> _nodeOffsets; // Armazena os offsets de início de cada nó no arquivo
        private bool _disposed = false;

        public ComputationGraphCacheManager(string sessionId)
        {
            var cacheDirectory = Path.Combine(Environment.CurrentDirectory, "Dayson", "TensorCache");
            Directory.CreateDirectory(cacheDirectory);
            _cacheFilePath = Path.Combine(cacheDirectory, $"{sessionId}.graphcache");
            _nodeOffsets = new List<long>();
            // O arquivo é aberto no Reset, antes do início de uma nova sequência
        }

        /// <summary>
        /// Limpa o cache anterior e prepara o arquivo para uma nova passagem forward.
        /// </summary>
        public void Reset()
        {
            _nodeOffsets.Clear();
            
            // Garante que streams antigos sejam fechados antes de truncar o arquivo
            _writer?.Close();
            _fileStream?.Close();

            // Abre o arquivo no modo Create, o que o trunca (zera) efetivamente
            _fileStream = new FileStream(_cacheFilePath, FileMode.Create, FileAccess.Write, FileShare.None);
            _writer = new BinaryWriter(_fileStream);
        }

        /// <summary>
        /// Serializa um único nó do grafo e o anexa ao arquivo de cache no disco.
        /// </summary>
        public void CacheNode(ComputationGraphNode node)
        {
            if (_writer == null || _fileStream == null)
                throw new InvalidOperationException("O cache não foi resetado antes do uso.");

            // Registra a posição atual, que é o início deste nó
            _nodeOffsets.Add(_fileStream.Position);

            // Escreve cada propriedade do nó no stream. Usar "" para nulos previne erros.
            _writer.Write(node.Timestep);
            _writer.Write(node.InputId ?? "");
            _writer.Write(node.HiddenPrevId ?? "");
            _writer.Write(node.CellPrevId ?? "");
            _writer.Write(node.ForgetGateId ?? "");
            _writer.Write(node.InputGateId ?? "");
            _writer.Write(node.CellCandidateId ?? "");
            _writer.Write(node.TanhCellNextId ?? "");
            _writer.Write(node.OutputGateId ?? "");
            _writer.Write(node.CellNextId ?? "");
            _writer.Write(node.HiddenNextId ?? "");
            _writer.Flush();
        }

        /// <summary>
        /// Retorna um iterador que lê os nós do grafo do disco em ordem REVERSA.
        /// Isso é extremamente eficiente em memória, pois apenas um nó é desserializado por vez.
        /// </summary>
        public IEnumerable<ComputationGraphNode> RetrieveNodesBackward()
        {
            // Fecha o stream de escrita para garantir que todos os dados estejam no disco
            _writer?.Close();
            _fileStream?.Close();
            
            // Abre um novo stream apenas para leitura
            using var readStream = new FileStream(_cacheFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(readStream);

            // Itera sobre os offsets de trás para frente
            for (int i = _nodeOffsets.Count - 1; i >= 0; i--)
            {
                long offset = _nodeOffsets[i];
                readStream.Seek(offset, SeekOrigin.Begin);

                var node = new ComputationGraphNode
                {
                    Timestep = reader.ReadInt32(),
                    InputId = reader.ReadString(),
                    HiddenPrevId = reader.ReadString(),
                    CellPrevId = reader.ReadString(),
                    ForgetGateId = reader.ReadString(),
                    InputGateId = reader.ReadString(),
                    CellCandidateId = reader.ReadString(),
                    TanhCellNextId = reader.ReadString(),
                    OutputGateId = reader.ReadString(),
                    CellNextId = reader.ReadString(),
                    HiddenNextId = reader.ReadString()
                };

                // 'yield return' transforma o método em um iterador (lazy evaluation)
                yield return node;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            _writer?.Dispose();
            _fileStream?.Dispose();

            try
            {
                if (File.Exists(_cacheFilePath))
                {
                    File.Delete(_cacheFilePath);
                }
            }
            catch (IOException)
            {
                // Ignora erros se o arquivo estiver bloqueado
            }
            _disposed = true;
        }
    }
}
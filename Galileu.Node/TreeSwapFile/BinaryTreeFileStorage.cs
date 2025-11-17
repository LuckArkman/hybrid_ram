using System;
using System.IO;
using System.Text;
using System.Threading;

namespace Galileu.Node.TreeSwapFile;

public class BinaryTreeFileStorage : IDisposable
{
    private readonly string _filePath;
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly BinaryReader _reader;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
    private bool _disposed = false;
    private static readonly Encoding Utf8NoBom = new UTF8Encoding(false);

    public BinaryTreeFileStorage(string filePath)
    {
        _filePath = filePath;
        var directory = Path.GetDirectoryName(_filePath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        _fileStream = new FileStream(_filePath, FileMode.Create, FileAccess.ReadWrite, FileShare.None);
        _writer = new BinaryWriter(_fileStream, Utf8NoBom, true);
        _reader = new BinaryReader(_fileStream, Utf8NoBom, true);
    }

    // CORREÇÃO COMPLETA - BinaryTreeFileStorage.cs
// Substitua o método StoreData existente por este

    public long StoreData(byte[] dataBytes)
    {
        if (dataBytes == null)
            throw new ArgumentNullException(nameof(dataBytes));
        if (dataBytes.Length == 0)
            throw new ArgumentException("Array de dados está vazio", nameof(dataBytes));

        _lock.EnterWriteLock();
        try
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));

            // ✅ Salva tamanho original para rollback
            long originalLength = _fileStream.Length;
            long offset = originalLength;

            // ✅ Verifica espaço em disco ANTES
            var drive = new DriveInfo(Path.GetPathRoot(_filePath));
            long requiredSpace = dataBytes.Length + sizeof(int) + sizeof(int) + (1024 * 1024); // +1MB buffer

            if (drive.AvailableFreeSpace < requiredSpace)
            {
                throw new IOException(
                    $"Espaço em disco insuficiente. " +
                    $"Necessário: {requiredSpace / (1024 * 1024)}MB, " +
                    $"Disponível: {drive.AvailableFreeSpace / (1024 * 1024)}MB");
            }

            try
            {
                _fileStream.Seek(offset, SeekOrigin.Begin);

                // Calcula checksum
                int checksum = CalculateChecksum(dataBytes);

                // Escreve dados
                _writer.Write(dataBytes.Length); // 4 bytes
                _writer.Write(checksum); // 4 bytes
                _writer.Write(dataBytes); // N bytes

                // Flush forçado
                _writer.Flush();
                _fileStream.Flush(flushToDisk: true);

                // ✅ Valida escrita completa
                long expectedPosition = offset + sizeof(int) + sizeof(int) + dataBytes.Length;
                if (_fileStream.Position != expectedPosition)
                {
                    throw new IOException(
                        $"Escrita incompleta. Esperado posição {expectedPosition}, atual {_fileStream.Position}");
                }

                return offset;
            }
            catch (Exception writeEx)
            {
                // ✅ CORREÇÃO: ROLLBACK automático
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[Storage] ERRO na escrita: {writeEx.Message}");
                Console.WriteLine($"[Storage] Executando ROLLBACK para {originalLength} bytes...");
                Console.ResetColor();

                try
                {
                    // Trunca arquivo para tamanho original
                    _fileStream.SetLength(originalLength);
                    _fileStream.Flush(flushToDisk: true);

                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"[Storage] ✓ Rollback executado com sucesso");
                    Console.ResetColor();
                }
                catch (Exception rollbackEx)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[Storage] ✗ CRÍTICO: Rollback falhou: {rollbackEx.Message}");
                    Console.WriteLine($"[Storage] Arquivo pode estar corrompido!");
                    Console.ResetColor();

                    // Lança exceção composta com ambos os erros
                    throw new InvalidOperationException(
                        $"Escrita falhou E rollback falhou. Sistema em estado inconsistente.\n" +
                        $"Erro original: {writeEx.Message}\n" +
                        $"Erro de rollback: {rollbackEx.Message}", writeEx);
                }

                // Re-lança exceção original após rollback bem-sucedido
                throw;
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Calcula checksum simples para validação de integridade
    /// </summary>
    private int CalculateChecksum(byte[] data)
    {
        unchecked
        {
            int checksum = 0;
            for (int i = 0; i < data.Length; i++)
            {
                checksum = (checksum * 31) + data[i];
            }

            return checksum;
        }
    }

    /// <summary>
    /// Lê dados com validação de checksum
    /// </summary>
    public byte[] GetDataBytes(long offset)
    {
        _lock.EnterReadLock();
        try
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));

            if (offset < 0 || offset >= _fileStream.Length)
                throw new ArgumentOutOfRangeException(nameof(offset),
                    $"Offset {offset} inválido. Tamanho do arquivo: {_fileStream.Length}");

            _fileStream.Seek(offset, SeekOrigin.Begin);

            int dataLength = _reader.ReadInt32();

            // ✅ Validação de tamanho
            if (dataLength < 0 || dataLength > 100_000_000) // 100MB limite
            {
                throw new InvalidDataException($"Tamanho de dados suspeito: {dataLength} bytes no offset {offset}");
            }

            int storedChecksum = _reader.ReadInt32(); // ✅ Lê checksum
            byte[] data = _reader.ReadBytes(dataLength);

            // ✅ Validação de integridade
            if (data.Length != dataLength)
            {
                throw new InvalidDataException(
                    $"Dados truncados. Esperado: {dataLength} bytes, lido: {data.Length} bytes");
            }

            // ✅ Valida checksum
            int calculatedChecksum = CalculateChecksum(data);
            if (calculatedChecksum != storedChecksum)
            {
                throw new InvalidDataException(
                    $"Checksum inválido no offset {offset}. " +
                    $"Esperado: {storedChecksum}, calculado: {calculatedChecksum}. " +
                    $"Dados podem estar corrompidos!");
            }

            return data;
        }
        catch (EndOfStreamException ex)
        {
            throw new InvalidDataException($"Fim inesperado do arquivo no offset {offset}", ex);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public long StoreData(string data)
    {
        byte[] dataBytes = Utf8NoBom.GetBytes(data);
        return StoreData(dataBytes);
    }

    public void Flush()
    {
        _lock.EnterWriteLock();
        try
        {
            _writer?.Flush();
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public string GetData(long offset)
    {
        _lock.EnterReadLock();
        try
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));
            _fileStream.Seek(offset, SeekOrigin.Begin);
            int dataLength = _reader.ReadInt32();
            byte[] dataBytes = _reader.ReadBytes(dataLength);
            return Utf8NoBom.GetString(dataBytes);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    // MÉTODO CORRIGIDO (ADICIONADO)
    // Este método público implementa a interface IDisposable.
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    // Este método protegido contém a lógica de limpeza.
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            _lock.EnterWriteLock();
            try
            {
                _writer?.Close();
                _reader?.Close();
                _fileStream?.Close();
            }
            finally
            {
                _lock.ExitWriteLock();
            }

            _lock?.Dispose();
        }

        _disposed = true;
    }

    public void Clear()
    {
        _lock.EnterWriteLock();
        try
        {
            if (_disposed) return;
            _fileStream.SetLength(0);
            _fileStream.Flush();
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
}
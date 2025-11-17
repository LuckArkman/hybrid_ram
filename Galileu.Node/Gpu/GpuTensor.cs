using System;
using System.Buffers;
using System.Linq;
using Galileu.Node.Interfaces;
using Galileu.Node.Core;
using OpenCL.NetCore;
using System.IO;
using System.Runtime.InteropServices;
using static OpenCL.NetCore.Cl;
using Exception = System.Exception;

namespace Galileu.Node.Gpu
{
    public class GpuTensor : IMathTensor
    {
        public int[] Shape { get; }
        public long Length { get; }
        public bool IsGpu => true;

        internal Mem Buffer { get; private set; }
        private static readonly ArrayPool<byte> IoBufferPool = ArrayPool<byte>.Shared;
        private static readonly ArrayPool<float> IoFloatBufferPool = ArrayPool<float>.Shared; // ✅ NOVO
        private readonly Context _context;
        private readonly CommandQueue _commandQueue;
        private readonly GpuSyncGuard _syncGuard; // ✅ NOVO: Guard de sincronização
        private bool _disposed = false;

        // ✅ NOVO: Rastreamento de operações pendentes
        private int _pendingOperations = 0;
        private readonly object _syncLock = new object();

        public GpuTensor(int[] shape, Context context, CommandQueue commandQueue, GpuSyncGuard syncGuard)
        {
            Shape = shape;
            Length = shape.Aggregate(1L, (a, b) => a * b);
            _context = context;
            _commandQueue = commandQueue;
            _syncGuard = syncGuard ?? throw new ArgumentNullException(nameof(syncGuard));

            ErrorCode error;
            Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite, (IntPtr)(Length * sizeof(float)), IntPtr.Zero,
                out error);
            if (error != ErrorCode.Success)
                throw new OpenClException("Falha ao alocar buffer do tensor.", error);
        }

        public GpuTensor(float[] data, int[] shape, Context context, CommandQueue commandQueue, GpuSyncGuard syncGuard)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (shape == null || shape.Length == 0) throw new ArgumentException("Shape inválido", nameof(shape));
            if (data.Any(val => float.IsNaN(val) || float.IsInfinity(val)))
                throw new ArgumentException("Dados contêm NaN/Infinity", nameof(data));

            Shape = shape;
            Length = data.Length;
            _context = context;
            _commandQueue = commandQueue;
            _syncGuard = syncGuard ?? throw new ArgumentNullException(nameof(syncGuard));

            ErrorCode error;
            Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                (IntPtr)(Length * sizeof(float)), data, out error);
            if (error != ErrorCode.Success)
                throw new OpenClException($"Falha ao alocar buffer GPU. Error: {error}", error);
        }

        /// <summary>
        /// ✅ CORRIGIDO: ToCpuTensor com sincronização ANTES da leitura.
        /// </summary>
        public Tensor ToCpuTensor()
        {
            lock (_syncLock)
            {
                // 🔒 CRÍTICO: Sincroniza ANTES de ler da GPU
                _syncGuard.SynchronizeBeforeRead($"ToCpuTensor[{GetTensorId()}]");

                var floatData = new float[Length];
                ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True,
                    IntPtr.Zero, (IntPtr)(Length * sizeof(float)), floatData, 0, null, out Event readEvent);

                if (error != ErrorCode.Success)
                    throw new OpenClException("Falha ao ler dados da GPU para a CPU.", error);

                // Aguarda o evento de leitura completar
                _syncGuard.WaitForEvent(readEvent, $"ToCpuRead[{GetTensorId()}]");

                return new Tensor(floatData, Shape);
            }
        }

        /// <summary>
        /// ✅ CORRIGIDO: UpdateFromCpu com sincronização ANTES da escrita.
        /// </summary>
        public void UpdateFromCpu(float[] data)
        {
            if (data.Length != this.Length)
                throw new ArgumentException("Tamanho dos dados incompatível.");

            lock (_syncLock)
            {
                // 🔒 CRÍTICO: Sincroniza ANTES de escrever na GPU
                _syncGuard.SynchronizeBeforeRead($"UpdateFromCpu[{GetTensorId()}]");

                ErrorCode error = EnqueueWriteBuffer(_commandQueue, Buffer, Bool.True,
                    IntPtr.Zero, (IntPtr)(Length * sizeof(float)), data, 0, null, out Event writeEvent);

                if (error != ErrorCode.Success)
                    throw new OpenClException("Falha ao escrever dados da CPU para a GPU.", error);

                // Aguarda escrita completar
                _syncGuard.WaitForEvent(writeEvent, $"UpdateWrite[{GetTensorId()}]");
            }
        }

        public unsafe void WriteToStream(BinaryWriter writer)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GpuTensor));

            int byteSize = (int)(Length * sizeof(float));
            if (byteSize == 0) return;

            // ✅ Aluga buffer da pool em vez de "new float[]"
            float[] hostBuffer = IoFloatBufferPool.Rent((int)Length);

            try
            {
                lock (_syncLock)
                {
                    _syncGuard.SynchronizeBeforeRead($"PreWriteToStream[{GetTensorId()}]");

                    GCHandle handle = GCHandle.Alloc(hostBuffer, GCHandleType.Pinned);
                    try
                    {
                        IntPtr hostPtr = handle.AddrOfPinnedObject();

                        ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True,
                            IntPtr.Zero, (IntPtr)byteSize, hostPtr, 0, null, out Event readEvent);

                        if (error != ErrorCode.Success)
                            throw new OpenClException($"Falha ao ler buffer da GPU para a RAM em WriteToStream.", error);

                        _syncGuard.WaitForEvent(readEvent, $"WriteToStreamRead[{GetTensorId()}]");

                        // Usa AsSpan para garantir que apenas os dados relevantes sejam escritos
                        var byteSpan = MemoryMarshal.AsBytes(hostBuffer.AsSpan(0, (int)Length));
                        writer.Write(byteSpan);
                        writer.Flush();
                    }
                    finally
                    {
                        if (handle.IsAllocated) handle.Free();
                    }
                }
            }
            finally
            {
                // ✅ CRÍTICO: Devolve o buffer para a pool, independentemente de sucesso ou falha.
                IoFloatBufferPool.Return(hostBuffer);
            }
        }

        /// <summary>
        /// ✅ VERSÃO CORRIGIDA FINAL: ReadFromStream com ZERO retenção.
        /// </summary>
        public unsafe void ReadFromStream(BinaryReader reader, long length)
        {
            if (length != Length)
                throw new InvalidDataException($"Tamanho inconsistente: {length} vs {Length}");

            int byteSize = (int)(length * sizeof(float));
            if (byteSize == 0) return;

            lock (_syncLock)
            {
                // 🔥 ESTRATÉGIA 1: Sync ANTES
                _syncGuard.SynchronizeBeforeRead($"PreRead[{GetTensorId()}]");
                Flush(_commandQueue);

                // 🔥 ESTRATÉGIA 2: Aloca buffer de bytes para leitura
                byte[] byteBuffer = new byte[byteSize];
                int bytesRead = reader.Read(byteBuffer, 0, byteSize);
                if (bytesRead != byteSize)
                    throw new EndOfStreamException($"Leitura incompleta: {bytesRead}/{byteSize}");

                // 🔥 ESTRATÉGIA 3: Converte bytes → float usando MemoryMarshal
                float[] hostBuffer = new float[length];
                MemoryMarshal.Cast<byte, float>(byteBuffer.AsSpan()).CopyTo(hostBuffer.AsSpan());

                // Libera buffer de bytes imediatamente
                byteBuffer = null;

                GCHandle handle = GCHandle.Alloc(hostBuffer, GCHandleType.Pinned);

                try
                {
                    IntPtr hostPtr = handle.AddrOfPinnedObject();

                    // 🔥 ESTRATÉGIA 4: Escrita BLOQUEANTE
                    ErrorCode error = EnqueueWriteBuffer(
                        _commandQueue,
                        Buffer,
                        Bool.True, // BLOCKING
                        IntPtr.Zero,
                        (IntPtr)byteSize,
                        hostPtr,
                        0,
                        null,
                        out Event writeEvent
                    );

                    if (error != ErrorCode.Success)
                        throw new OpenClException("Falha ao escrever GPU", error);

                    // 🔥 ESTRATÉGIA 5: Wait + Release imediato
                    WaitForEvents(1, new[] { writeEvent });
                    ReleaseEvent(writeEvent);

                    // 🔥 ESTRATÉGIA 6: Finish pós-escrita
                    Finish(_commandQueue);
                }
                finally
                {
                    // 🔥 ESTRATÉGIA 7: Cleanup total
                    if (handle.IsAllocated) handle.Free();
                    hostBuffer = null;
                    GC.Collect(0, GCCollectionMode.Forced, false);
                }
            }
        }

        /// <summary>
        /// ✅ CORRIGIDO: Dispose com sincronização ANTES de liberar buffer.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;

            // Delega a lógica de liberação para o novo método.
            ReleaseResources();

            // Apenas marca como 'disposed' e suprime o finalizador APÓS
            // a liberação bem-sucedida.
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// ✅ CORRIGIDO: Finalizer com logging e sync.
        /// </summary>
        ~GpuTensor()
        {
            if (!_disposed)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[GpuTensor] 🔴 VAZAMENTO DETECTADO: Tensor [{string.Join("x", Shape)}] não foi descartado explicitamente. Stack Trace da criação pode estar no log do GpuMemoryTracker.");
                Console.ResetColor();
        
                // Tenta liberar os recursos como último recurso.
                // Isso pode ou não funcionar dependendo do estado da aplicação.
                try
                {
                    ReleaseResources();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[GpuTensor] 🔴 Falha ao tentar liberar tensor vazado durante a finalização: {ex.Message}");
                }
            }
        }
        
        private void ReleaseResources()
        {
            // Este método NÃO deve ter um bloco try-catch. Se a sincronização ou
            // a liberação falhar, é uma condição de erro crítica e o programa
            // deve ser interrompido para evitar corrupção de estado.
            lock (_syncLock)
            {
                if (!Buffer.Equals(default(Mem)))
                {
                    long sizeBytes = Length * sizeof(float);
                    _syncGuard.SynchronizeBeforeDispose(GetTensorId(), sizeBytes);

                    var error = ReleaseMemObject(Buffer);
                    if (error != ErrorCode.Success)
                    {
                        // Lança uma exceção que irá parar a aplicação. Este é o
                        // comportamento correto quando não podemos garantir o estado da VRAM.
                        throw new OpenClException($"CRÍTICO: Falha ao liberar buffer de VRAM para o tensor {GetTensorId()}", error);
                    }
            
                    // Marca o buffer como liberado para evitar dupla liberação.
                    Buffer = default(Mem);
                }
            }
        }

        // Helper para logging
        private string GetTensorId() => $"{string.Join("x", Shape)}_{GetHashCode():X8}";
    }
}
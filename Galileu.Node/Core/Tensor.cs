using System;
using System.Buffers;
using System.IO;
using System.Runtime.InteropServices;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Core
{
    public class Tensor : IMathTensor
    {
        private readonly float[] data;
        public int[] shape { get; } // Tornando a propriedade pública para consistência com a interface
        
        private static readonly ArrayPool<byte> IoBufferPool = ArrayPool<byte>.Shared;

        public Tensor(float[] data, int[] shape)
        {
            this.data = data ?? throw new ArgumentNullException(nameof(data));
            this.shape = shape ?? throw new ArgumentNullException(nameof(shape));

            // ✅ CORRIGIDO: Inicializa a propriedade Length.
            this.Length = data.Length;

            long expectedSize = 1;
            foreach (int dim in shape)
            {
                if (dim <= 0)
                {
                    throw new ArgumentException("As dimensões do shape devem ser positivas.");
                }
                // Usamos long para evitar overflow em tensores muito grandes
                expectedSize *= dim;
            }

            if (data.Length != expectedSize)
            {
                throw new ArgumentException(
                    $"O tamanho dos dados ({data.Length}) não corresponde às dimensões do shape ({string.Join("x", shape)}), esperado {expectedSize}.");
            }
        }

        // ... (método Infer permanece o mesmo) ...
        public float Infer(int[] indices)
        {
            if (indices == null || indices.Length != shape.Length)
                throw new ArgumentException("Os índices fornecidos não correspondem às dimensões do tensor.");

            for (int i = 0; i < indices.Length; i++)
            {
                if (indices[i] < 0 || indices[i] >= shape[i])
                    throw new ArgumentOutOfRangeException(nameof(indices), $"Índice {indices[i]} fora dos limites para a dimensão {i} (0 a {shape[i] - 1}).");
            }

            int flatIndex = 0;
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                flatIndex += indices[i] * stride;
                stride *= shape[i];
            }

            return data[flatIndex];
        }


        public float[] GetData() => (float[])data.Clone();
        
        // Implementação explícita da interface IMathTensor
        int[] IMathTensor.Shape => (int[])this.shape.Clone();
        public long Length { get; }

        Tensor IMathTensor.ToCpuTensor()
        {
            return this;
        }

        public void UpdateFromCpu(float[] newData)
        {
            if (newData == null) throw new ArgumentNullException(nameof(newData));
            if (newData.Length != this.data.Length) throw new ArgumentException("Tamanho dos dados incompatível.");
            Array.Copy(newData, this.data, newData.Length);
        }

        /// <summary>
        /// Escreve o conteúdo do tensor de forma eficiente para um stream.
        /// </summary>
        public void WriteToStream(BinaryWriter writer)
        {
            // ✅ CORRETO: Operação "zero-copy" altamente eficiente.
            var byteData = MemoryMarshal.AsBytes(data.AsSpan());
            writer.Write(byteData);
        }
        
        /// <summary>
        /// ✅ ATUALIZADO: Lê dados de um stream de forma eficiente usando ArrayPool.
        /// </summary>
        public void ReadFromStream(BinaryReader reader, long length)
        {
            if (length != this.Length)
            {
                throw new InvalidDataException($"Inconsistência no tamanho do tensor. Esperado pelo objeto: {this.Length}, informado pelo gerenciador: {length}.");
            }
                
            int bytesToRead = (int)length * sizeof(float);
            if (bytesToRead == 0) return;

            // Aluga um buffer reutilizável para evitar sobrecarregar o GC.
            byte[] rentedBuffer = IoBufferPool.Rent(bytesToRead);
            
            try
            {
                int totalBytesRead = reader.Read(rentedBuffer, 0, bytesToRead);
                if (totalBytesRead != bytesToRead)
                {
                    throw new EndOfStreamException($"Não foi possível ler a quantidade esperada de bytes ({bytesToRead}) do stream. Bytes lidos: {totalBytesRead}.");
                }
                
                // Copia do buffer reutilizado para o array de dados final do tensor.
                MemoryMarshal.Cast<byte, float>(rentedBuffer.AsSpan(0, bytesToRead)).CopyTo(this.data);
            }
            finally
            {
                // Devolve o buffer ao pool para ser reutilizado.
                IoBufferPool.Return(rentedBuffer);
            }
        }

        public bool IsGpu => false;

        public void Dispose() { /* Nenhum recurso não gerenciado para liberar. */ }
    }
}
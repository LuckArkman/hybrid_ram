using System;
using System.Buffers;
using System.IO;
using System.Runtime.InteropServices;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Cpu
{
    public class CpuTensor : IMathTensor
    {
        private readonly float[] _data;
        public int[] Shape { get; }
        public long Length { get; }
        public bool IsGpu => false;
        
        private static readonly ArrayPool<byte> IoBufferPool = ArrayPool<byte>.Shared;

        public CpuTensor(float[] data, int[] shape)
        {
            _data = data;
            Shape = shape;
            Length = data.Length;
        }

        public float[] GetData() => _data;

        public Tensor ToCpuTensor() => new Tensor((float[])_data.Clone(), Shape);

        public void UpdateFromCpu(float[] data)
        {
            if (data.Length != this.Length)
            {
                throw new ArgumentException("Os dados de entrada devem ter o mesmo tamanho do tensor.");
            }
            Array.Copy(data, _data, this.Length);
        }

        /// <summary>
        /// Escreve o conteúdo do tensor da RAM para um stream binário (arquivo).
        /// Esta operação é altamente eficiente ("zero-copy"), pois cria uma visualização
        /// somente leitura dos dados como bytes e os escreve diretamente no stream.
        /// </summary>
        public void WriteToStream(BinaryWriter writer)
        {
            var byteData = MemoryMarshal.AsBytes(_data.AsSpan());
            writer.Write(byteData);
        }
        
        /// <summary>
        /// Lê dados de um stream binário (arquivo) e os carrega na memória deste tensor de CPU.
        /// Utiliza um buffer de RAM intermediário e reutilizável do ArrayPool para a leitura,
        /// evitando alocações de memória desnecessárias.
        /// </summary>
        public void ReadFromStream(BinaryReader reader, long length)
        {
            if (length != this.Length)
            {
                throw new InvalidDataException($"Inconsistência no tamanho do tensor. Esperado pelo objeto: {this.Length}, informado pelo gerenciador: {length}.");
            }
            
            int bytesToRead = (int)length * sizeof(float);
            if (bytesToRead == 0) return;

            byte[] rentedBuffer = IoBufferPool.Rent(bytesToRead);
        
            try
            {
                int totalBytesRead = reader.Read(rentedBuffer, 0, bytesToRead);
                if (totalBytesRead != bytesToRead)
                {
                    throw new EndOfStreamException($"Não foi possível ler a quantidade esperada de bytes ({bytesToRead}) do stream. Bytes lidos: {totalBytesRead}.");
                }
            
                MemoryMarshal.Cast<byte, float>(rentedBuffer.AsSpan(0, bytesToRead)).CopyTo(_data);
            }
            finally
            {
                IoBufferPool.Return(rentedBuffer);
            }
        }
        
        public void Dispose()
        {
            // Nada a fazer para este tensor de CPU, pois não possui recursos não gerenciados.
        }
    }
}
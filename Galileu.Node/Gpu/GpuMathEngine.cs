using Galileu.Node.Interfaces;
using OpenCL.NetCore;
using System;
using System.Linq;
using static OpenCL.NetCore.Cl;
using Exception = System.Exception;

namespace Galileu.Node.Gpu;

public class GpuMathEngine : IMathEngine, IDisposable
{
    public bool IsGpu => true;
    private readonly Context _context;
    private readonly CommandQueue _commandQueue;
    private readonly OpenCL.NetCore.Program _program;
    private readonly bool _isXeonCpu;
    private readonly GpuSyncGuard _syncGuard;
    private readonly EventPool _eventPool;
    
    private int _operationsSinceLastSync = 0;
    private const int SYNC_INTERVAL = 100;

    // Kernels
    private readonly Kernel _matrixMultiplyKernel;
    private readonly Kernel _vectorMatrixMultiplyKernel;
    private readonly Kernel _addKernel;
    private readonly Kernel _addBroadcastKernel;
    private readonly Kernel _multiplyKernel;
    private readonly Kernel _sigmoidKernel;
    private readonly Kernel _tanhKernel;
    private readonly Kernel _cloneKernel;
    private readonly Kernel _transposeKernel;
    private readonly Kernel _subtractKernel;
    private readonly Kernel _sigmoidDerivativeKernel;
    private readonly Kernel _tanhDerivativeKernel;
    private readonly Kernel _matrixMultiplyTransposeAKernel;
    private readonly Kernel _matrixMultiplyTransposeBKernel;
    private readonly Kernel _addScaledKernel;
    private readonly Kernel _subtractScaledKernel;
    private readonly Kernel _sliceKernel;
    private readonly Kernel _setKernel;
    private readonly Kernel _clipKernel;
    private readonly Kernel _scaleKernel;
    private readonly Kernel _softmaxKernel;
    private readonly Kernel _lookupKernel;
    private readonly Kernel _accumulateGradientKernel;
    private readonly Kernel _oneHotEncodeKernel;
    private readonly Kernel _adamUpdateKernel;
    private readonly Kernel _layerNormKernel;
    private readonly Kernel _sanitizeAndClipKernel;
    private readonly Kernel _sumOfSquaresKernel;

    private bool _disposed = false;

    #region Kernels Source - ULTRA ROBUSTO FINAL
    private const string ProgramSource = @"
__kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, int M, int N, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < N; ++k) { sum += A[i * N + k] * B[k * P + j]; } C[i * P + j] = sum; } }
__kernel void vector_matrix_multiply(__global const float* V, __global const float* M, __global float* R, int N, int P) { int j = get_global_id(0); if (j < P) { float sum = 0.0f; for (int k = 0; k < N; ++k) { sum += V[k] * M[k * P + j]; } R[j] = sum; } }
__kernel void add(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] + b[gid]; }
__kernel void add_broadcast(__global float* a, __global const float* bias, int bias_size) { int gid = get_global_id(0); int col = gid % bias_size; if (col < bias_size && gid < get_global_size(0)) { a[gid] = a[gid] + bias[col]; } }
__kernel void multiply(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] * b[gid]; }
__kernel void clone_buffer(__global const float* input, __global float* output) { int gid = get_global_id(0); output[gid] = input[gid]; }
__kernel void transpose(__global const float* input, __global float* output, int rows, int cols) { int i = get_global_id(0); int j = get_global_id(1); if (i < rows && j < cols) { output[j * rows + i] = input[i * cols + j]; } }
__kernel void subtract(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] - b[gid]; }
__kernel void matrix_multiply_transpose_a(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[k * M + i] * B[k * P + j]; } C[i * P + j] = sum; } }
__kernel void matrix_multiply_transpose_b(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[i * K + k] * B[j * K + k]; } C[i * P + j] = sum; } }
__kernel void add_scaled(__global float* target, __global const float* source, float scalar) { int gid = get_global_id(0); target[gid] += source[gid] * scalar; }
__kernel void subtract_scaled(__global float* target, __global const float* source, float scalar) { int gid = get_global_id(0); target[gid] -= source[gid] * scalar; }
__kernel void slice(__global const float* source, __global float* dest, int offset, int size) { int gid = get_global_id(0); if (gid < size) { dest[gid] = source[offset + gid]; } }
__kernel void set(__global float* dest, __global const float* source, int offset, int size) { int gid = get_global_id(0); if (gid < size) { dest[offset + gid] = source[gid]; } }
__kernel void clip(__global float* data, float min_val, float max_val) { int gid = get_global_id(0); data[gid] = fmax(min_val, fmin(max_val, data[gid])); }
__kernel void scale(__global float* data, float scalar) { int gid = get_global_id(0); data[gid] *= scalar; }
__kernel void lookup(__global const float* embedding_matrix, __global float* result, int index, int embedding_size) { int gid = get_global_id(0); if (gid < embedding_size) { result[gid] = embedding_matrix[index * embedding_size + gid]; } }
__kernel void accumulate_gradient_no_atomic(__global float* embedding_gradients, __global const float* gradient, int index, int embedding_size) { int gid = get_global_id(0); if (gid < embedding_size) { embedding_gradients[index * embedding_size + gid] += gradient[gid]; } }
__kernel void one_hot_encode(__global float* output, __global const int* indices, int total_classes) { int i = get_global_id(0); int row_offset = i * total_classes; for(int j = 0; j < total_classes; ++j) { output[row_offset + j] = 0.0f; } int one_hot_index = indices[i]; output[row_offset + one_hot_index] = 1.0f; }
__kernel void tanh_activation(__global const float* a, __global float* result) { int gid = get_global_id(0); float input = a[gid]; const float MAX_TANH_INPUT = 20.0f; input = clamp(input, -MAX_TANH_INPUT, MAX_TANH_INPUT); if (isnan(input) || isinf(input)) { result[gid] = 0.0f; return; } float exp2x = exp(2.0f * input); if (isinf(exp2x) || isnan(exp2x)) { result[gid] = (input > 0.0f) ? 1.0f : -1.0f; return; } float tanh_result = (exp2x - 1.0f) / (exp2x + 1.0f); if (isnan(tanh_result) || isinf(tanh_result)) { result[gid] = (input > 0.0f) ? 1.0f : -1.0f; } else { result[gid] = clamp(tanh_result, -1.0f, 1.0f); } }
__kernel void sigmoid(__global const float* a, __global float* result) { int gid = get_global_id(0); float input = a[gid]; const float MAX_SIGMOID_INPUT = 88.0f; input = clamp(input, -MAX_SIGMOID_INPUT, MAX_SIGMOID_INPUT); if (isnan(input) || isinf(input)) { result[gid] = 0.5f; return; } float sigmoid_result; if (input >= 0.0f) { float exp_neg = exp(-input); if (isinf(exp_neg) || isnan(exp_neg)) { sigmoid_result = 1.0f; } else { sigmoid_result = 1.0f / (1.0f + exp_neg); } } else { float exp_pos = exp(input); if (isinf(exp_pos) || isnan(exp_pos)) { sigmoid_result = 0.0f; } else { sigmoid_result = exp_pos / (1.0f + exp_pos); } } if (isnan(sigmoid_result) || isinf(sigmoid_result)) { result[gid] = 0.5f; } else { result[gid] = clamp(sigmoid_result, 0.0f, 1.0f); } }
__kernel void tanh_derivative(__global const float* output, __global float* result) { int gid = get_global_id(0); float o = output[gid]; if (isnan(o) || isinf(o)) { result[gid] = 0.0f; return; } float deriv = 1.0f - o * o; if (isnan(deriv) || isinf(deriv)) { result[gid] = 0.0f; } else { result[gid] = clamp(deriv, 0.0f, 1.0f); } }
__kernel void sigmoid_derivative(__global const float* output, __global float* result) { int gid = get_global_id(0); float o = output[gid]; if (isnan(o) || isinf(o)) { result[gid] = 0.0f; return; } float deriv = o * (1.0f - o); if (isnan(deriv) || isinf(deriv)) { result[gid] = 0.0f; } else { result[gid] = clamp(deriv, 0.0f, 0.25f); } }
__kernel void softmax(__global const float* input, __global float* output, int size) { int row = get_global_id(0); int offset = row * size; float maxVal = input[offset]; for (int i = 1; i < size; i++) { float val = input[offset + i]; if (!isnan(val) && !isinf(val) && val > maxVal) { maxVal = val; } } maxVal = clamp(maxVal, -88.0f, 88.0f); float sumExp = 0.0f; for (int i = 0; i < size; i++) { float val = input[offset + i]; if (isnan(val) || isinf(val)) { output[offset + i] = 0.0f; continue; } float shifted = clamp(val - maxVal, -88.0f, 0.0f); float exp_val = exp(shifted); if (isnan(exp_val) || isinf(exp_val)) { exp_val = 0.0f; } output[offset + i] = exp_val; sumExp += exp_val; } if (sumExp < 1e-10f || isnan(sumExp) || isinf(sumExp)) { float uniform = 1.0f / (float)size; for (int i = 0; i < size; i++) { output[offset + i] = uniform; } } else { for (int i = 0; i < size; i++) { float normalized = output[offset + i] / sumExp; if (isnan(normalized) || isinf(normalized)) { output[offset + i] = 1e-10f; } else { output[offset + i] = clamp(normalized, 1e-10f, 1.0f); } } } }
__kernel void adam_update(__global float* p, __global const float* g, __global float* m, __global float* v, float lr, float beta1, float beta2, float epsilon, int t) { int i = get_global_id(0); float grad = g[i]; if (isnan(grad) || isinf(grad)) return; const float MAX_GRAD = 10.0f; grad = clamp(grad, -MAX_GRAD, MAX_GRAD); float m_old = m[i]; float v_old = v[i]; if (isnan(m_old) || isinf(m_old)) m_old = 0.0f; if (isnan(v_old) || isinf(v_old)) v_old = 0.0f; float m_val = beta1 * m_old + (1.0f - beta1) * grad; float v_val = beta2 * v_old + (1.0f - beta2) * (grad * grad); if (isnan(m_val) || isinf(m_val)) m_val = 0.0f; if (isnan(v_val) || isinf(v_val)) v_val = 0.0f; float beta1_pow_t = pow(beta1, (float)t); float beta2_pow_t = pow(beta2, (float)t); float m_hat = m_val / (1.0f - beta1_pow_t); float v_hat = v_val / (1.0f - beta2_pow_t); if (isnan(m_hat) || isinf(m_hat)) m_hat = m_val; if (isnan(v_hat) || isinf(v_hat)) v_hat = v_val; float sqrt_v_hat = sqrt(fabs(v_hat)); float denominator = sqrt_v_hat + epsilon; if (denominator < 1e-8f) return; float update = lr * m_hat / denominator; const float MAX_UPDATE = 0.1f; update = clamp(update, -MAX_UPDATE, MAX_UPDATE); if (isnan(update) || isinf(update)) return; p[i] -= update; m[i] = m_val; v[i] = v_val; }
__kernel void layer_norm(__global float* input, __global const float* gamma, __global const float* beta, int size, float epsilon) { int row = get_global_id(0); int offset = row * size; float mean = 0.0f; for (int i = 0; i < size; ++i) { mean += input[offset + i]; } mean /= size; float variance = 0.0f; for (int i = 0; i < size; ++i) { float diff = input[offset + i] - mean; variance += diff * diff; } variance /= size; float inv_std = rsqrt(variance + epsilon); for (int i = 0; i < size; ++i) { input[offset + i] = ((input[offset + i] - mean) * inv_std) * gamma[i] + beta[i]; } }

__kernel void sanitize_and_clip(__global float* data, float clip_val) {
    int gid = get_global_id(0);
    float current_val = data[gid];
    if (isnan(current_val) || isinf(current_val)) {
        data[gid] = 0.0f;
    } else {
        data[gid] = clamp(current_val, -clip_val, clip_val);
    }
}

__kernel void sum_of_squares(__global const float* input, __local float* scratch, __global float* result, uint length) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    
    float local_sum = 0.0f;
    for (int i = global_id; i < length; i += get_global_size(0)) {
        local_sum += input[i] * input[i];
    }
    
    scratch[local_id] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            scratch[local_id] += scratch[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}
";
    #endregion

    public GpuMathEngine()
    {
        ErrorCode error;
        Platform[] platforms = GetPlatformIDs(out error); CheckError(error);
        var platform = platforms.First();
        Device[] devices = GetDeviceIDs(platform, DeviceType.Gpu, out error);
        if (error != ErrorCode.Success || devices.Length == 0)
        {
            devices = GetDeviceIDs(platform, DeviceType.Cpu, out error); CheckError(error);
        }
        var device = devices[0];
        
        string deviceName = GetDeviceInfo(device, DeviceInfo.Name, out error).ToString();
        _isXeonCpu = deviceName.ToLower().Contains("xeon");
        
        _context = CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error); CheckError(error);
        _commandQueue = CreateCommandQueue(_context, device, CommandQueueProperties.None, out error); CheckError(error);
        _program = CreateProgramWithSource(_context, 1, new[] { ProgramSource }, null, out error); CheckError(error);
        
        string buildOptions = _isXeonCpu 
            ? "-cl-single-precision-constant -cl-denorms-are-zero -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -cl-mad-enable" 
            : "-cl-fast-relaxed-math -cl-mad-enable";
        
        error = BuildProgram(_program, 1, new[] { device }, buildOptions, null, IntPtr.Zero);
        if (error != ErrorCode.Success)
        {
            string buildLog = GetProgramBuildInfo(_program, device, ProgramBuildInfo.Log, out _).ToString();
            throw new OpenClException($"Erro ao compilar kernels: {buildLog}", error);
        }

        _matrixMultiplyKernel = CreateKernel(_program, "matrix_multiply", out error); CheckError(error);
        _vectorMatrixMultiplyKernel = CreateKernel(_program, "vector_matrix_multiply", out error); CheckError(error);
        _addKernel = CreateKernel(_program, "add", out error); CheckError(error);
        _addBroadcastKernel = CreateKernel(_program, "add_broadcast", out error); CheckError(error);
        _multiplyKernel = CreateKernel(_program, "multiply", out error); CheckError(error);
        _sigmoidKernel = CreateKernel(_program, "sigmoid", out error); CheckError(error);
        _tanhKernel = CreateKernel(_program, "tanh_activation", out error); CheckError(error);
        _cloneKernel = CreateKernel(_program, "clone_buffer", out error); CheckError(error);
        _transposeKernel = CreateKernel(_program, "transpose", out error); CheckError(error);
        _subtractKernel = CreateKernel(_program, "subtract", out error); CheckError(error);
        _sigmoidDerivativeKernel = CreateKernel(_program, "sigmoid_derivative", out error); CheckError(error);
        _tanhDerivativeKernel = CreateKernel(_program, "tanh_derivative", out error); CheckError(error);
        _matrixMultiplyTransposeAKernel = CreateKernel(_program, "matrix_multiply_transpose_a", out error); CheckError(error);
        _matrixMultiplyTransposeBKernel = CreateKernel(_program, "matrix_multiply_transpose_b", out error); CheckError(error);
        _addScaledKernel = CreateKernel(_program, "add_scaled", out error); CheckError(error);
        _subtractScaledKernel = CreateKernel(_program, "subtract_scaled", out error); CheckError(error);
        _sliceKernel = CreateKernel(_program, "slice", out error); CheckError(error);
        _setKernel = CreateKernel(_program, "set", out error); CheckError(error);
        _clipKernel = CreateKernel(_program, "clip", out error); CheckError(error);
        _scaleKernel = CreateKernel(_program, "scale", out error); CheckError(error);
        _softmaxKernel = CreateKernel(_program, "softmax", out error); CheckError(error);
        _lookupKernel = CreateKernel(_program, "lookup", out error); CheckError(error);
        _accumulateGradientKernel = CreateKernel(_program, "accumulate_gradient_no_atomic", out error); CheckError(error);
        _oneHotEncodeKernel = CreateKernel(_program, "one_hot_encode", out error); CheckError(error);
        _adamUpdateKernel = CreateKernel(_program, "adam_update", out error); CheckError(error);
        _layerNormKernel = CreateKernel(_program, "layer_norm", out error); CheckError(error);
        _sanitizeAndClipKernel = CreateKernel(_program, "sanitize_and_clip", out error); CheckError(error);
        _sumOfSquaresKernel = CreateKernel(_program, "sum_of_squares", out error); CheckError(error);

        _syncGuard = new GpuSyncGuard(_commandQueue);
        _eventPool = new EventPool();
    }

    public void Synchronize()
    {
        _syncGuard.SynchronizeBeforeRead("ManualSync");
        _operationsSinceLastSync = 0;
    }

    public IMathTensor CreateTensor(int[] shape) => new GpuTensor(shape, _context, _commandQueue, _syncGuard);
    
    public IMathTensor CreateTensor(float[] data, int[] shape)
    {
        long expectedLength = shape.Aggregate(1L, (a, b) => a * b);
        if (data.Length != expectedLength)
            throw new ArgumentException($"Tamanho incompatível: {data.Length} vs {expectedLength}");
        return new GpuTensor(data, shape, _context, _commandQueue, _syncGuard);
    }

    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a; var tensorB = (GpuTensor)b;
        int M = tensorA.Shape[0]; int N = tensorA.Shape[1]; int P = tensorB.Shape[1];
        ExecuteKernel2D(_matrixMultiplyKernel, M, P, tensorA, tensorB, (GpuTensor)result, M, N, P);
        CheckPeriodicSync("MatrixMultiply");
    }
    
    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        ExecuteKernel1D(_addKernel, a.Length, (GpuTensor)a, (GpuTensor)b, (GpuTensor)result);
        CheckPeriodicSync("Add");
    }

    public void AddBroadcast(IMathTensor a, IMathTensor bias, IMathTensor result)
    {
        if (!ReferenceEquals(a, result)) { ExecuteKernel1D(_cloneKernel, a.Length, (GpuTensor)a, (GpuTensor)result); }
        ExecuteKernel1D(_addBroadcastKernel, result.Length, (GpuTensor)result, (GpuTensor)bias, (int)bias.Length);
    }
    
    private void CheckPeriodicSync(string operationName)
    {
        _operationsSinceLastSync++;
        if (_operationsSinceLastSync >= SYNC_INTERVAL)
        {
            //Console.WriteLine($"[GpuMathEngine] 🔄 Sync periódico após {_operationsSinceLastSync} ops");
            _syncGuard.SynchronizeBeforeRead($"PeriodicSync_{operationName}");
            _eventPool.Cleanup();
            _operationsSinceLastSync = 0;
        }
    }

    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result) => ExecuteKernel1D(_multiplyKernel, a.Length, (GpuTensor)a, (GpuTensor)b, (GpuTensor)result);
    public void Sigmoid(IMathTensor a, IMathTensor result) => ExecuteKernel1D(_sigmoidKernel, a.Length, (GpuTensor)a, (GpuTensor)result);
    public void Tanh(IMathTensor a, IMathTensor result) => ExecuteKernel1D(_tanhKernel, a.Length, (GpuTensor)a, (GpuTensor)result);

    public IMathTensor Clone(IMathTensor tensor)
    {
        var newTensor = CreateTensor(tensor.Shape) as GpuTensor;
        ExecuteKernel1D(_cloneKernel, tensor.Length, (GpuTensor)tensor, newTensor);
        return newTensor;
    }

    public void Transpose(IMathTensor input, IMathTensor result)
    {
        int rows = input.Shape[0]; int cols = input.Shape[1];
        ExecuteKernel2D(_transposeKernel, rows, cols, (GpuTensor)input, (GpuTensor)result, rows, cols);
    }

    public void Subtract(IMathTensor a, IMathTensor b, IMathTensor result) => ExecuteKernel1D(_subtractKernel, a.Length, (GpuTensor)a, (GpuTensor)b, (GpuTensor)result);
    public void SigmoidDerivative(IMathTensor output, IMathTensor result) => ExecuteKernel1D(_sigmoidDerivativeKernel, output.Length, (GpuTensor)output, (GpuTensor)result);
    public void TanhDerivative(IMathTensor output, IMathTensor result) => ExecuteKernel1D(_tanhDerivativeKernel, output.Length, (GpuTensor)output, (GpuTensor)result);

    public void MatrixMultiplyTransposeA(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a; var tensorB = (GpuTensor)b;
        int M = tensorA.Shape[1]; int K = tensorA.Shape[0]; int P = tensorB.Shape[1];
        ExecuteKernel2D(_matrixMultiplyTransposeAKernel, M, P, tensorA, tensorB, (GpuTensor)result, M, K, P);
    }

    public void MatrixMultiplyTransposeB(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a; var tensorB = (GpuTensor)b;
        int M = tensorA.Shape[0]; int K = tensorA.Shape[1]; int P = tensorB.Shape[0];
        ExecuteKernel2D(_matrixMultiplyTransposeBKernel, M, P, tensorA, tensorB, (GpuTensor)result, M, K, P);
    }

    public void AddScaled(IMathTensor target, IMathTensor source, float scalar) => ExecuteKernel1D(_addScaledKernel, target.Length, (GpuTensor)target, (GpuTensor)source, scalar);
    public void SubtractScaled(IMathTensor target, IMathTensor source, float scalar) => ExecuteKernel1D(_subtractScaledKernel, target.Length, (GpuTensor)target, (GpuTensor)source, scalar);

    public void Slice(IMathTensor source, int rowIndex, IMathTensor destination)
    {
        var featureSize = (int)destination.Length; var offset = rowIndex * featureSize;
        ExecuteKernel1D(_sliceKernel, featureSize, (GpuTensor)source, (GpuTensor)destination, offset, featureSize);
    }

    public void Set(IMathTensor destination, int rowIndex, IMathTensor source)
    {
        var featureSize = (int)source.Length; var offset = rowIndex * featureSize;
        ExecuteKernel1D(_setKernel, featureSize, (GpuTensor)destination, (GpuTensor)source, offset, featureSize);
    }

    public void Clip(IMathTensor tensor, float minValue, float maxValue) => ExecuteKernel1D(_clipKernel, tensor.Length, (GpuTensor)tensor, minValue, maxValue);
    public void Scale(IMathTensor tensor, float scalar) => ExecuteKernel1D(_scaleKernel, tensor.Length, (GpuTensor)tensor, scalar);

    public void Softmax(IMathTensor input, IMathTensor result)
    {
        int rows = input.Shape[0]; int cols = input.Shape[1];
        ExecuteKernel1D(_softmaxKernel, rows, (GpuTensor)input, (GpuTensor)result, cols);
    }

    public void Lookup(IMathTensor embeddingMatrix, int index, IMathTensor result)
    {
        var embeddingSize = embeddingMatrix.Shape[1];
        ExecuteKernel1D(_lookupKernel, embeddingSize, (GpuTensor)embeddingMatrix, (GpuTensor)result, index, embeddingSize);
    }

    public void AccumulateGradient(IMathTensor embeddingGradients, IMathTensor gradient, int index)
    {
        var embeddingSize = embeddingGradients.Shape[1];
        ExecuteKernel1D(_accumulateGradientKernel, embeddingSize, (GpuTensor)embeddingGradients, (GpuTensor)gradient, index, embeddingSize);
    }

    public IMathTensor CreateOneHotTensor(int[] indices, int totalClasses)
    {
        int sequenceLength = indices.Length;
        var resultTensor = CreateTensor(new[] { sequenceLength, totalClasses }) as GpuTensor;
        Mem indicesBuffer = default;
        try
        {
            ErrorCode error;
            indicesBuffer = (Mem)CreateBuffer(_context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, (IntPtr)(indices.Length * sizeof(int)), indices, out error);
            CheckError(error);
            ExecuteKernel1D(_oneHotEncodeKernel, sequenceLength, resultTensor, indicesBuffer, totalClasses);
        }
        finally
        {
            // ✅ CORREÇÃO: A chamada para ReleaseMemObject é segura mesmo com um handle default/nulo.
            ReleaseMemObject(indicesBuffer);
        }
        return resultTensor;
    }

    public void AdamUpdate(IMathTensor parameters, IMathTensor gradients, IMathTensor m, IMathTensor v, float lr, float beta1, float beta2, float epsilon, int t)
    {
        ExecuteKernel1D(_adamUpdateKernel, parameters.Length, (GpuTensor)parameters, (GpuTensor)gradients, (GpuTensor)m, (GpuTensor)v, lr, beta1, beta2, epsilon, t);
    }

    public void LayerNorm(IMathTensor input, IMathTensor gamma, IMathTensor beta, float epsilon = 1e-5f)
    {
        int rows = input.Shape[0]; int size = input.Shape[1];
        ExecuteKernel1D(_layerNormKernel, rows, (GpuTensor)input, (GpuTensor)gamma, (GpuTensor)beta, size, epsilon);
    }

    public void SanitizeAndClip(IMathTensor tensor, float clipValue)
    {
        ExecuteKernel1D(_sanitizeAndClipKernel, tensor.Length, (GpuTensor)tensor, clipValue);
    }

    public double CalculateSumOfSquares(IMathTensor tensor)
    {
        if (tensor.Length == 0) return 0.0;

        const int workGroupSize = 256;
        int numWorkGroups = (int)((tensor.Length + workGroupSize - 1) / workGroupSize);
        numWorkGroups = Math.Max(1, numWorkGroups);
        long globalSize = numWorkGroups * workGroupSize;

        using (var partialResultBuffer = new GpuTensor(new[] { numWorkGroups }, _context, _commandQueue, _syncGuard))
        {
            SetKernelArg(_sumOfSquaresKernel, 0, ((GpuTensor)tensor).Buffer);
            SetKernelArg(_sumOfSquaresKernel, 1, (IntPtr)(workGroupSize * sizeof(float)), null);
            SetKernelArg(_sumOfSquaresKernel, 2, partialResultBuffer.Buffer);
            SetKernelArg(_sumOfSquaresKernel, 3, (uint)tensor.Length);

            ErrorCode error = EnqueueNDRangeKernel(_commandQueue, _sumOfSquaresKernel, 1, null, new[] { (IntPtr)globalSize }, new[] { (IntPtr)workGroupSize }, 0, null, out Event kernelEvent);
            CheckError(error, "Falha ao executar kernel sum_of_squares");
            
            try { WaitForEvents(1, new[] { kernelEvent }); } finally { ReleaseEvent(kernelEvent); }

            float[] partialSums = partialResultBuffer.ToCpuTensor().GetData();
            double totalSum = 0.0;
            for (int i = 0; i < partialSums.Length; i++) { totalSum += partialSums[i]; }
            return totalSum;
        }
    }

    private void ExecuteKernel1D(Kernel kernel, long globalSize, params object[] args)
    {
        ExecuteKernelWithCleanup(kernel, 1, new[] { (IntPtr)globalSize }, args);
    }

    private void ExecuteKernel2D(Kernel kernel, long globalSizeX, long globalSizeY, params object[] args)
    {
        ExecuteKernelWithCleanup(kernel, 2, new[] { (IntPtr)globalSizeX, (IntPtr)globalSizeY }, args);
    }
    
    private void ExecuteKernelWithCleanup(Kernel kernel, int workDim, IntPtr[] globalSize, params object[] args)
    {
        SetKernelArgs(kernel, args);
        ErrorCode error = EnqueueNDRangeKernel(_commandQueue, kernel, (uint)workDim, null, globalSize, null, 0, null, out Event kernelEvent);
        CheckError(error, $"Falha ao executar kernel");
        try { WaitForEvents(1, new[] { kernelEvent }); } finally { ReleaseEvent(kernelEvent); }
    }

    private void SetKernelArgs(Kernel kernel, params object[] args)
    {
        ErrorCode error;
        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            error = arg switch
            {
                GpuTensor tensor => SetKernelArg(kernel, (uint)i, tensor.Buffer),
                Mem memBuffer => SetKernelArg(kernel, (uint)i, memBuffer),
                int intVal => SetKernelArg(kernel, (uint)i, (uint)intVal),
                uint uintVal => SetKernelArg(kernel, (uint)i, uintVal),
                float floatVal => SetKernelArg(kernel, (uint)i, floatVal),
                _ => throw new ArgumentException($"Unsupported kernel argument type: {arg?.GetType().Name ?? "null"} at index {i}.")
            };
            CheckError(error, $"Failed to set kernel argument at index {i} for kernel '{GetKernelInfo(kernel, KernelInfo.FunctionName, out _)}'.");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        //Console.WriteLine("[GpuMathEngine] 🧹 Iniciando cleanup...");
        try
        {
            Synchronize();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuMathEngine] ⚠️ Erro no sync final: {ex.Message}");
        }
    
        ReleaseKernel(_matrixMultiplyKernel); ReleaseKernel(_vectorMatrixMultiplyKernel); ReleaseKernel(_addKernel); ReleaseKernel(_addBroadcastKernel);
        ReleaseKernel(_multiplyKernel); ReleaseKernel(_sigmoidKernel); ReleaseKernel(_tanhKernel); ReleaseKernel(_cloneKernel);
        ReleaseKernel(_transposeKernel); ReleaseKernel(_subtractKernel); ReleaseKernel(_sigmoidDerivativeKernel); ReleaseKernel(_tanhDerivativeKernel);
        ReleaseKernel(_matrixMultiplyTransposeAKernel); ReleaseKernel(_matrixMultiplyTransposeBKernel); ReleaseKernel(_addScaledKernel); ReleaseKernel(_subtractScaledKernel);
        ReleaseKernel(_sliceKernel); ReleaseKernel(_setKernel); ReleaseKernel(_clipKernel); ReleaseKernel(_scaleKernel);
        ReleaseKernel(_softmaxKernel); ReleaseKernel(_lookupKernel); ReleaseKernel(_accumulateGradientKernel); ReleaseKernel(_oneHotEncodeKernel);
        ReleaseKernel(_adamUpdateKernel); ReleaseKernel(_layerNormKernel);
        ReleaseKernel(_sanitizeAndClipKernel);
        ReleaseKernel(_sumOfSquaresKernel);
    
        ReleaseProgram(_program);
        ReleaseCommandQueue(_commandQueue);
        ReleaseContext(_context);
    
        _eventPool?.Dispose();
        _syncGuard?.Dispose();
    
        _disposed = true;
        GC.SuppressFinalize(this);
        //Console.WriteLine("[GpuMathEngine] ✅ Cleanup concluído");
    }

    ~GpuMathEngine() { Dispose(); }
    
    public void FlushQueue()
    {
        ErrorCode error = Flush(_commandQueue);
        if (error != ErrorCode.Success)
        {
            Console.WriteLine($"[GpuMathEngine] ⚠️ Erro ao fazer flush: {error}");
        }
    }

    private void CheckError(ErrorCode error, string message = "Erro OpenCL")
    {
        if (error != ErrorCode.Success) { throw new OpenClException($"{message} (Código: {error})", error); }
    }

    public GpuSyncGuard? GetSyncGuard() => _syncGuard;
}
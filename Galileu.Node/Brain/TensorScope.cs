using System;
using System.Collections.Generic;
using System.Diagnostics;
using Galileu.Node.Interfaces;
using Galileu.Node.Gpu;

namespace Galileu.Node.Brain;

/// <summary>
/// ✅ VERSÃO CORRIGIDA: TensorScope com sincronização automática no dispose.
/// Previne retorno de tensores ao pool com operações GPU pendentes.
/// </summary>
public class TensorScope : IDisposable
{
    private readonly List<IMathTensor> _managedTensors;
    private readonly List<Action> _cleanupActions;
    private readonly Stopwatch _stopwatch;
    private bool _disposed = false;

    private readonly IMathEngine _mathEngine;
    private readonly IndividualFileTensorManager? _tensorManager;
    private readonly TensorPool? _pool;
    private readonly GpuSyncGuard? _syncGuard;  // ✅ NOVO: Referência ao sync guard
    
    private readonly string _scopeName;

    /// <summary>
    /// ✅ CORRIGIDO: Construtor agora aceita SyncGuard opcional.
    /// </summary>
    public TensorScope(string scopeName, IMathEngine mathEngine, 
        IndividualFileTensorManager? tensorManager = null, TensorPool? pool = null)
    {
        _scopeName = scopeName;
        _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
        _tensorManager = tensorManager;
        _pool = pool;
        
        // ✅ NOVO: Obtém SyncGuard se for GPU engine
        if (mathEngine is GpuMathEngine gpuEngine)
        {
            _syncGuard = gpuEngine.GetSyncGuard();
        }

        _managedTensors = new List<IMathTensor>();
        _cleanupActions = new List<Action>();
        _stopwatch = Stopwatch.StartNew();
    }
    
    // ============================================================================
    // MÉTODOS DE FÁBRICA (mantém implementação existente)
    // ============================================================================
    
    public IMathTensor CreateTensor(int[] shape)
    {
        var tensor = _mathEngine.CreateTensor(shape);
        return Track(tensor);
    }

    public IMathTensor CreateTensor(float[] data, int[] shape)
    {
        var tensor = _mathEngine.CreateTensor(data, shape);
        return Track(tensor);
    }

    public IMathTensor LoadTensor(string id)
    {
        if (_tensorManager == null)
            throw new InvalidOperationException("TensorManager não fornecido ao TensorScope.");

        var tensor = _tensorManager.LoadTensor(id);
        return Track(tensor);
    }
    
    public TensorScope CreateSubScope(string name)
    {
        return new TensorScope($"{_scopeName}.{name}", _mathEngine, _tensorManager, _pool);
    }

    public T Track<T>(T tensor) where T : IMathTensor
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorScope));
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        _managedTensors.Add(tensor);
        return tensor;
    }
    
    public void Track(Action cleanupAction)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorScope));
        if (cleanupAction == null) throw new ArgumentNullException(nameof(cleanupAction));
        _cleanupActions.Add(cleanupAction);
    }
    
    public T Untrack<T>(T tensor) where T : IMathTensor
    {
        _managedTensors.Remove(tensor);
        return tensor;
    }
    
    // ============================================================================
    // ✅ DISPOSE CORRIGIDO COM SINCRONIZAÇÃO
    // ============================================================================
    
    public void Dispose()
    {
        if (_disposed) return;
        _stopwatch.Stop();
        
        // 🔒 SYNC POINT 1: Antes de qualquer cleanup
        if (_syncGuard != null)
        {
            try
            {
                _syncGuard.SynchronizeBeforeRead($"ScopeDispose_{_scopeName}");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[TensorScope:{_scopeName}] ⚠️ Erro ao sincronizar antes de dispose: {ex.Message}");
                Console.ResetColor();
                // Continua o dispose mesmo com falha de sync
            }
        }
        
        // Executa cleanup actions em ordem reversa
        for (int i = _cleanupActions.Count - 1; i >= 0; i--)
        {
            try 
            { 
                _cleanupActions[i].Invoke(); 
            }
            catch (Exception ex) 
            { 
                Console.WriteLine($"[TensorScope:{_scopeName}] ⚠️ Erro na ação de limpeza: {ex.Message}"); 
            }
        }
        _cleanupActions.Clear();

        // ✅ CORRIGIDO: Libera tensores com sync garantido
        for (int i = _managedTensors.Count - 1; i >= 0; i--)
        {
            var tensor = _managedTensors[i];
            try
            {
                if (_pool != null && tensor.IsGpu)
                {
                    // 🔒 CRÍTICO: Não retorna ao pool se há operações pendentes
                    // O GpuTensor.Dispose() já faz sync interno
                    _pool.Return(tensor);
                }
                else
                {
                    tensor.Dispose();
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[TensorScope:{_scopeName}] 🔴 Erro ao liberar tensor: {ex.Message}");
                Console.ResetColor();
            }
        }
        _managedTensors.Clear();
        
        // 🔒 SYNC POINT 2: Após dispose para garantir que tudo foi liberado
        if (_syncGuard != null)
        {
            try
            {
                _syncGuard.SynchronizeBeforeRead($"PostDispose_{_scopeName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TensorScope:{_scopeName}] ⚠️ Erro ao sincronizar após dispose: {ex.Message}");
            }
        }
        
        _disposed = true;
        
        // Log de performance (apenas para scopes importantes)
        if (_stopwatch.ElapsedMilliseconds > 1000)
        {
            //Console.WriteLine($"[TensorScope:{_scopeName}] ⏱️ Duração: {_stopwatch.ElapsedMilliseconds}ms");
        }
    }
}
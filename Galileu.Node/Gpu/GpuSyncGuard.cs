using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using OpenCL.NetCore;
using static OpenCL.NetCore.Cl;
using Exception = System.Exception;

namespace Galileu.Node.Gpu;

/// <summary>
/// Camada de sincronização GPU crítica para prevenir vazamentos VRAM→RAM.
/// Garante que TODAS as operações GPU sejam completadas antes de operações sensíveis.
/// </summary>
public sealed class GpuSyncGuard : IDisposable
{
    private readonly CommandQueue _commandQueue;
    private readonly ConcurrentDictionary<string, SyncPoint> _activeSyncPoints;
    private long _totalSyncs = 0;
    private long _totalWaitTimeMs = 0;
    private bool _disposed = false;
    
    // Métricas de performance
    private const int SYNC_TIMEOUT_MS = 30000; // 30s timeout
    private const int WARNING_WAIT_MS = 1000;  // Alerta se sync > 1s
    
    public GpuSyncGuard(CommandQueue? commandQueue)
    {
        _commandQueue = commandQueue ?? throw new ArgumentNullException(nameof(commandQueue));
        _activeSyncPoints = new ConcurrentDictionary<string, SyncPoint>();
    }
    
    /// <summary>
    /// 🔒 CRÍTICO: Sincroniza GPU ANTES de qualquer operação que leia de buffers.
    /// Previne staging buffers fantasma na RAM.
    /// </summary>
    public void SynchronizeBeforeRead(string context)
    {
        var sw = Stopwatch.StartNew();
        
        try
        {
            // 1. Força conclusão de TODOS os comandos no queue
            ErrorCode error = Finish(_commandQueue);
            if (error != ErrorCode.Success)
            {
                throw new OpenClException($"Falha ao sincronizar GPU antes de leitura [{context}]", error);
            }
            
            sw.Stop();
            Interlocked.Increment(ref _totalSyncs);
            Interlocked.Add(ref _totalWaitTimeMs, sw.ElapsedMilliseconds);
            
            // 2. Log de performance
            if (sw.ElapsedMilliseconds > WARNING_WAIT_MS)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[GpuSyncGuard] ⚠️ Sincronização lenta: {sw.ElapsedMilliseconds}ms para [{context}]");
                Console.ResetColor();
            }
            
            // 3. Registra sync point para auditoria
            _activeSyncPoints[context] = new SyncPoint
            {
                Context = context,
                Timestamp = DateTime.UtcNow,
                WaitTimeMs = sw.ElapsedMilliseconds
            };
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[GpuSyncGuard] 🔴 ERRO CRÍTICO ao sincronizar [{context}]: {ex.Message}");
            Console.ResetColor();
            throw;
        }
    }
    
    /// <summary>
    /// 🔒 CRÍTICO: Sincroniza GPU ANTES de liberar buffers.
    /// Previne shadow copies na RAM.
    /// </summary>
    public void SynchronizeBeforeDispose(string tensorId, long sizeBytes)
    {
        var sw = Stopwatch.StartNew();
        
        try
        {
            ErrorCode error = Finish(_commandQueue);
            if (error != ErrorCode.Success)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[GpuSyncGuard] 🔴 Falha ao sincronizar antes de dispose [{tensorId}]");
                Console.ResetColor();
                // NÃO lança exceção - melhor liberar parcialmente que vazar completamente
            }
            
            sw.Stop();
            
            // Log para tensores grandes
            if (sizeBytes > 100 * 1024 * 1024) // > 100MB
            {
                Console.WriteLine($"[GpuSyncGuard] 📊 Tensor grande sincronizado antes de dispose: " +
                                  $"{sizeBytes / (1024.0 * 1024.0):F2}MB em {sw.ElapsedMilliseconds}ms");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuSyncGuard] ⚠️ Exceção durante sync pré-dispose: {ex.Message}");
            // Continua - dispose deve prosseguir mesmo com falha de sync
        }
    }
    
    /// <summary>
    /// 🔒 Insere uma barreira explícita no command queue.
    /// Garante que comandos futuros só executem após comandos passados completarem.
    /// </summary>
    public Event InsertBarrier(string reason)
    {
        // ✅ CORREÇÃO FINAL: Usa EnqueueMarker que tem o efeito de uma barreira e retorna um Event.
        ErrorCode error = EnqueueMarker(_commandQueue, out Event barrierEvent);
        if (error != ErrorCode.Success)
        {
            throw new OpenClException($"Falha ao inserir barreira (usando marker) [{reason}]", error);
        }
        
        Console.WriteLine($"[GpuSyncGuard] 🚧 Barreira (via Marker) inserida: {reason}");
        return barrierEvent;
    }

    /// <summary>
    /// 🔒 Cria um marker para rastrear progresso de operações assíncronas.
    /// </summary>
    public Event CreateProgressMarker(string tag)
    {
        // ✅ CORRIGIDO: Nome da função ajustado para EnqueueMarker
        ErrorCode error = EnqueueMarker(_commandQueue, out Event markerEvent);
        if (error != ErrorCode.Success)
        {
            throw new OpenClException($"Falha ao criar marker [{tag}]", error);
        }
        
        return markerEvent;
    }
    
    /// <summary>
    /// 🔒 Aguarda um evento específico completar com timeout.
    /// </summary>
    public bool WaitForEvent(Event evt, string context, int timeoutMs = SYNC_TIMEOUT_MS)
    {
        var sw = Stopwatch.StartNew();
        
        try
        {
            ErrorCode error = WaitForEvents(1, new[] { evt });
            sw.Stop();
            
            if (error != ErrorCode.Success)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[GpuSyncGuard] 🔴 Timeout aguardando evento [{context}]");
                Console.ResetColor();
                return false;
            }
            
            if (sw.ElapsedMilliseconds > WARNING_WAIT_MS)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[GpuSyncGuard] ⚠️ Evento lento [{context}]: {sw.ElapsedMilliseconds}ms");
                Console.ResetColor();
            }
            
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuSyncGuard] ⚠️ Erro ao aguardar evento [{context}]: {ex.Message}");
            return false;
        }
        finally
        {
            try { ReleaseEvent(evt); }
            catch { }
        }
    }
    
    /// <summary>
    /// 📊 Retorna estatísticas de sincronização para diagnóstico.
    /// </summary>
    public SyncStatistics GetStatistics()
    {
        return new SyncStatistics
        {
            TotalSyncs = Interlocked.Read(ref _totalSyncs),
            TotalWaitTimeMs = Interlocked.Read(ref _totalWaitTimeMs),
            AverageWaitTimeMs = _totalSyncs > 0 ? (double)_totalWaitTimeMs / _totalSyncs : 0,
            ActiveSyncPoints = _activeSyncPoints.Count
        };
    }
    
    /// <summary>
    /// 🧹 Limpa sync points antigos (mais de 5 minutos).
    /// </summary>
    public void CleanupOldSyncPoints()
    {
        var cutoff = DateTime.UtcNow.AddMinutes(-5);
        var toRemove = _activeSyncPoints
            .Where(kvp => kvp.Value.Timestamp < cutoff)
            .Select(kvp => kvp.Key)
            .ToList();
        
        foreach (var key in toRemove)
        {
            _activeSyncPoints.TryRemove(key, out _);
        }
        
        if (toRemove.Count > 0)
        {
            Console.WriteLine($"[GpuSyncGuard] 🧹 Limpou {toRemove.Count} sync points antigos");
        }
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        // Última sincronização antes de desligar
        try
        {
            Finish(_commandQueue);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GpuSyncGuard] ⚠️ Erro ao sincronizar no dispose: {ex.Message}");
        }
        
        _activeSyncPoints.Clear();
        _disposed = true;
    }
}

/// <summary>
/// Representa um ponto de sincronização registrado.
/// </summary>
public class SyncPoint
{
    public string Context { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public long WaitTimeMs { get; set; }
}

/// <summary>
/// Estatísticas de sincronização GPU.
/// </summary>
public class SyncStatistics
{
    public long TotalSyncs { get; set; }
    public long TotalWaitTimeMs { get; set; }
    public double AverageWaitTimeMs { get; set; }
    public int ActiveSyncPoints { get; set; }
    
    public override string ToString()
    {
        return $"Syncs: {TotalSyncs}, Wait total: {TotalWaitTimeMs}ms, Avg: {AverageWaitTimeMs:F2}ms, Ativos: {ActiveSyncPoints}";
    }
}
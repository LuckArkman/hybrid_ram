using Galileu.Node.Interfaces;
using OpenCL.NetCore;
using System;
using System.Linq;
using static OpenCL.NetCore.Cl;

namespace Galileu.Node.Gpu;

/// <summary>
/// ✅ NOVO: Pool de eventos OpenCL reutilizáveis.
/// </summary>
public class EventPool : IDisposable
{
    private readonly Stack<Event> _availableEvents = new Stack<Event>();
    private readonly HashSet<Event> _inUseEvents = new HashSet<Event>();
    private readonly object _lock = new object();

    public Event Rent()
    {
        lock (_lock)
        {
            if (_availableEvents.Count > 0)
            {
                var evt = _availableEvents.Pop();
                _inUseEvents.Add(evt);
                return evt;
            }

            // Não cria novos eventos - usa default
            return default(Event);
        }
    }

    public void Return(Event evt)
    {
        if (evt.Equals(default(Event))) return;

        lock (_lock)
        {
            _inUseEvents.Remove(evt);
            _availableEvents.Push(evt);
        }
    }

    public void Cleanup()
    {
        lock (_lock)
        {
            // Libera eventos não usados
            while (_availableEvents.Count > 10) // Mantém no máximo 10
            {
                var evt = _availableEvents.Pop();
                try
                {
                    ReleaseEvent(evt);
                }
                catch
                {
                }
            }
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            foreach (var evt in _availableEvents)
            {
                try
                {
                    ReleaseEvent(evt);
                }
                catch
                {
                }
            }

            _availableEvents.Clear();

            foreach (var evt in _inUseEvents)
            {
                try
                {
                    ReleaseEvent(evt);
                }
                catch
                {
                }
            }

            _inUseEvents.Clear();
        }
    }
}
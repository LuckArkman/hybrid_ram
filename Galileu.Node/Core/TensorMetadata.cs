namespace Galileu.Node.Core;

public class TensorMetadata
{
    public string Id { get; set; } = string.Empty;
    public long Offset { get; set; }
    public string FilePath { get; set; } = string.Empty;
    public int[] Shape { get; set; } = Array.Empty<int>();
    public DateTime CreatedAt { get; set; }
    public DateTime LastAccessed { get; set; }
    public long SizeBytes { get; set; }
    public int AccessCount;
}
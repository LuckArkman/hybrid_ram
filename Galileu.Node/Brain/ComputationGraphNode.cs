namespace Galileu.Node.Brain;

[Serializable]
public class ComputationGraphNode
{
    public int Timestep { get; set; }
    public string InputId { get; set; }
    public string HiddenPrevId { get; set; }
    public string CellPrevId { get; set; }
    public string ForgetGateId { get; set; }
    public string InputGateId { get; set; }
    public string CellCandidateId { get; set; }
    public string TanhCellNextId { get; set; }
    public string OutputGateId { get; set; }
    public string CellNextId { get; set; }
    public string HiddenNextId { get; set; }
}
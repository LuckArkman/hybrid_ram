using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public record ModelWeights
{
    // Pesos Principais
    public IMathTensor Embedding { get; init; }
    public IMathTensor W_if { get; init; }
    public IMathTensor W_hf { get; init; }
    public IMathTensor B_f { get; init; }
    public IMathTensor W_ii { get; init; }
    public IMathTensor W_hi { get; init; }
    public IMathTensor B_i { get; init; }
    public IMathTensor W_ic { get; init; }
    public IMathTensor W_hc { get; init; }
    public IMathTensor B_c { get; init; }
    public IMathTensor W_io { get; init; }
    public IMathTensor W_ho { get; init; }
    public IMathTensor B_o { get; init; }
    public IMathTensor W_hy { get; init; }
    public IMathTensor B_y { get; init; }

    // Pesos de Layer Normalization
    public IMathTensor LN_f_gamma { get; init; }
    public IMathTensor LN_f_beta { get; init; }
    public IMathTensor LN_i_gamma { get; init; }
    public IMathTensor LN_i_beta { get; init; }
    public IMathTensor LN_c_gamma { get; init; }
    public IMathTensor LN_c_beta { get; init; }
    public IMathTensor LN_o_gamma { get; init; }
    public IMathTensor LN_o_beta { get; init; }
}
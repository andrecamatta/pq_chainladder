module ChainLadderCredito

include("ScrCalibration.jl")
include("ScrModalidades.jl")
include("VintageSim.jl")
include("ChainLadderCore.jl")
include("Backtest.jl")
include("ApcModel.jl")
include("FatorComum.jl")

using .ScrCalibration
using .ScrModalidades
using .VintageSim
using .ChainLadderCore
using .Backtest
using .ApcModel
using .FatorComum

end # module

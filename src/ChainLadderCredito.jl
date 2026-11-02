module ChainLadderCredito

include("ScrCalibration.jl")
include("VintageSim.jl")
include("ChainLadderCore.jl")
include("Backtest.jl")

using .ScrCalibration
using .VintageSim
using .ChainLadderCore
using .Backtest

end # module

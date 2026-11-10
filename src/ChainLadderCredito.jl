module ChainLadderCredito

include("ScrCalibration.jl")
include("VintageSim.jl")
include("ChainLadderCore.jl")
include("Backtest.jl")
include("ApcModel.jl")

using .ScrCalibration
using .VintageSim
using .ChainLadderCore
using .Backtest
using .ApcModel

end # module

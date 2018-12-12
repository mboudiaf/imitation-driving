using Base.Test
using AutoEnvs
using JLD

function runtests()
#    include("test_debug_envs.jl")
#    println("TEST DEBUG ENVS PASSED")

#    include("test_ngsim_utils.jl")
#    println("TEST NGSIM UTILS PASSED")

#    include("test_ngsim_env.jl")
#    println("TEST NGSIM ENV PASSED")

#    include("test_vectorized_ngsim_env.jl")
#    println("TEST VECTORIZED NGSIM ENV PASSED")

    include("test_multiagent_ngsim_env.jl")
    println("All tests passed")
end

@time runtests()

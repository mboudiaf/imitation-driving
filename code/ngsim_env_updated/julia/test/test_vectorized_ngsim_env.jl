using Base.Test
using AutoEnvs

function test_basics()
    # ctor
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i80_trajectories-0400-0415.txt")
    n_envs = 100
    params = Dict("trajectory_filepaths"=>[filepath], "n_envs"=>n_envs)
    env = VectorizedNGSIMEnv(params)
    println("VECTORIZED ENV CREATED")
    # reset, step
    x = reset(env)
    a = zeros(n_envs, 2)
    nx, r, terminal, infos = step(env, a)
    @test x != nx
    nnx, r, terminal, infos = step(env, a)
    @test nx != nnx
    println("RESET VECTORIZED ENV DONE")
    # obs spec
    shape, spacetype, infos = observation_space_spec(env)
    
    @test spacetype == "Box"
    @test in("high", keys(infos))
    @test in("low", keys(infos))
    for i in 1:200
	_, _, terminals, _ = step(env, a)
        reset(env, terminals)
	println(i)
	 
   end
    
end

@time test_basics()

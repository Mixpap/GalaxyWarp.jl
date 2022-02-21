using GalaxyWarp
using Test


@testset "1D Signal Fitting" begin
    vv=collect(-400.0:20.0:400.0)

    test_parss=[[1.2,-100.0,22.0],
        [1.2,-100.0,22.0,  2.0,  11.0,30.0],
        [0.5,-100.0,22.0,  0.6,-170.0,26.0,  2.0,11.0,30.0],
        [0.5,-100.0,22.0,  0.5,-130.0,16.0,  2.0,11.0,30.0]
    ]
    for test_pars in test_parss
        test_signal=GalaxyWarp.gmodel(vv,test_pars)
        result_pars = GalaxyWarp.fit_pixel(test_signal,0.1,vv,0.1;smin=0.8,smax=100.0,svmin=15,svmax=34.0)[1]

        AA=test_pars[1:3:end]
        VVc=test_pars[2:3:end]
        SSv=test_pars[3:3:end]

        fAA=result_pars[1:3:end]
        fVVc=result_pars[2:3:end]
        fSSv=result_pars[3:3:end]
        @testset "$(size(AA)[1]) gaussians" begin
            @test isapprox(size(AA)[1] ,size(fAA)[1], atol=1)
            if size(AA)[1] == size(fAA)[1]
                    @test isapprox(minimum(abs.(sort(AA) .- sort(fAA))), 0.0, atol=0.5)
                    @test isapprox(minimum(abs.(sort(VVc) .- sort(fVVc))), 0.0, atol=15.0)
                    @test isapprox(minimum(abs.(sort(SSv) .- sort(fSSv))), 0.0, atol=10.0)
            end
        end
    end

    for test_pars in test_parss
        for rms in [0.1,0.15,0.2]
            test_signal=GalaxyWarp.gmodel(vv,test_pars)
            test_signal=rand.(GalaxyWarp.Normal.(test_signal,rms))
            result_pars = GalaxyWarp.fit_pixel(test_signal,rms,vv,rms;smin=0.8,smax=100.0,svmin=15,svmax=34.0)[1]

            AA=test_pars[1:3:end]
            VVc=test_pars[2:3:end]
            SSv=test_pars[3:3:end]

            fAA=result_pars[1:3:end]
            fVVc=result_pars[2:3:end]
            fSSv=result_pars[3:3:end]
            @testset "$(size(AA)[1]) gaussians" begin
                @test isapprox(size(AA)[1] ,size(fAA)[1], atol=1)
                if size(AA)[1] == size(fAA)[1]
                        @test isapprox(minimum(abs.(sort(AA) .- sort(fAA))), 0.0, atol=0.5)
                        @test isapprox(minimum(abs.(sort(VVc) .- sort(fVVc))), 0.0, atol=15.0)
                        @test isapprox(minimum(abs.(sort(SSv) .- sort(fSSv))), 0.0, atol=10.0)
                end
            end
        end
    end
end

using Parameters
using PCHIPInterpolation
using Optim
using BlackBoxOptim
using GLMakie 

@with_kw mutable struct p_test
    Fit::Array{Symbol}=[ 	
        :pa0,:pa15,:pa5,
        :i0,:i15,:i5,
        :v0,:a,:n]
    
    pa0::Float64=0.0
    pa15::Float64=20.0
    pa5::Float64=30.0
    
    PA::Array{Symbol}=[:pa0,:pa15,:pa5]
    R_pa::Vector{Float64}=[0.0,1.5,5.0]
    ppa = Interpolator
    
    i0::Float64=70.0
    i15::Float64=74.0
    i5::Float64=70.0
    
    I::Array{Symbol}=[:i0,:i15,:i5]
    R_i::Vector{Float64}=[0.0,1.5,5.0]
    ppi = Interpolator
    
    v0::Float64=300.0
    a::Float64= 2.0
    n::Float64= 1.0
    
    drc=0.05
    Rd=collect(0.01:drc:5.0)
    dR =0.07
    dV0=300.0
    dv=32.0
    V::Array{Symbol}=[:v0,:a,:n]

    Vcirc::Function = (r::Float64,v0::Float64,a::Float64,n::Float64) -> v0*r/(r^2.0+a^2.0)^n
    
end

@testset "Simulated data Fitting" begin
    simdata=GalaxyWarp.sim_data("$(@__DIR__)/test";theta_dist="uniform", Vcirc = x-> 320.0*x/(x^2.0+1.5^2.0)^1.2,I=x->70.0,PA=x->6.0*x, Sb = x->100.0, sigma_inst=x->10.0, cloud_mass=1e5 ,xl=3.0,yl=5.0,vl=350.0,dx=0.05,dy=0.05,dv=20.0,beam=[0.08,0.07],Rmin=0.1,dR=0.1,Rmax = 4.5,a=0.01,rms=0.05,Rcmax=2.1,Rcmap=1.7,k=3)

    @test length(simdata["Xc"])>100000
    @test length(simdata["Yc"])==length(simdata["Xc"])
    @test length(simdata["Vc"])==length(simdata["Xc"])
    @test length(simdata["Zc"])==length(simdata["Xc"])

    clouds_sim=GalaxyWarp.fit_pixelst(simdata;x0=-5.0,x1=5.0,y0=-5.5,y1=5.5,sigma=3.0, ncrit=1,N0=1,Nmax=7,vmin=-400.0, vmax=400.0,svmin=15.0, svmax=32.0,maxsteps=800, popsize=16)

    GalaxyWarp.cloud_fitting_diagnostics(simdata,clouds_sim,"$(@__DIR__)/test")

    lenX0=length(clouds_sim.Xp)
    @test lenX0>2000
    @test length(clouds_sim.Yp)==lenX0
    @test length(clouds_sim.Vp)==lenX0
    @test length(clouds_sim.belongs)==0
    @test length(clouds_sim.belongs)==0

    clouds_sim_f1=GalaxyWarp.filter_clouds(clouds_sim,[Dict("x0"=>-1.0)])

    lenX1=length(clouds_sim_f1.Xp)
    @test lenX1<lenX0
    @test length(clouds_sim_f1.Vp)==lenX1
    @test length(clouds_sim_f1.Yp)==lenX1

    @time GalaxyWarp.merge_pixel_clouds!(clouds_sim;dr=0.1,dv=50.0,mp=7,df=3.0*simdata["rms"],roll_d=8,max_d=1.0)

    GalaxyWarp.cloud_fitting_diagnostics(simdata,clouds_sim_f1,"$(@__DIR__)/test")


    lenX1m=length(clouds_sim.Xc)
    @test lenX1m>10
    @test lenX1m<lenX1
    @test length(clouds_sim.Vc)==lenX1m
    @test length(clouds_sim.Yc)==lenX1m

    Pars=p_test()
    par0=copy([getfield(Pars,s) for s in Pars.Fit])
    GalaxyWarp.update_parameters!(Pars,par0)
    disks0 = GalaxyWarp.make_disks(Pars.Rd,Pars)

    function logposterior(pars::Vector{Float64})::Float64
		GalaxyWarp.update_parameters!(Pars,pars)
		likedV=GalaxyWarp.likelihood(clouds_sim,Pars;dv=30.0)
		#prior=GalaxyWarp.logprior(Pars,Priors)
		return likedV
	end

    #res =optimize(logposterior, par0, NelderMead(),Optim.Options(iterations = 3600, store_trace = true, show_trace = true,g_tol=1.0e-3))

	#MAP=Optim.minimizer(res)

    plower=[0.0,0.0,0.0,
    0.0,50.0,50.0,
    50.0,0.5,0.3]

    pupper=[180.0,180.0,50.0,
    90.0,90.0,90.0,
    500.0,4.0,3.0]

    steps=10000
    opt_L = bbsetup(logposterior;
		SearchRange = collect(zip(plower, pupper)),
		PopulationSize=120,MaxSteps = steps
		)
	res_L = bboptimize(opt_L)
	MAP=best_candidate(res_L)

	@show MAP
    @show logposterior(MAP)

    GalaxyWarp.update_parameters!(Pars,MAP)
    disks = GalaxyWarp.make_disks(Pars.Rd,Pars)
	GalaxyWarp.cloud_geometry!(clouds_sim,disks,Pars)

    fsky=GalaxyWarp.plot_sky(clouds_sim,disks,msize=7)
	save("$(@__DIR__)/sky.png",fsky)

    pa_real0=0.0
    pa_real15=6.0*1.5
    pa_real5=6.0*5.0

    i_real0=70.0
    i_real15=70.0
    i_real5=70.0

    v0_real=320.0
    a_real=1.5
    n_real=1.2

    dpa=7.0
    di=7.0
    @test isapprox(pa_real0,Pars.pa0,atol=dpa)
    @test isapprox(pa_real15,Pars.pa15,atol=dpa)
    @test isapprox(pa_real5,Pars.pa5,atol=dpa)
    @test isapprox(i_real0,Pars.i0,atol=di)
    @test isapprox(i_real15,Pars.i15,atol=di)
    @test isapprox(i_real5,Pars.i5,atol=di)
    @test isapprox(v0_real,Pars.v0,atol=20.0 )
    @test isapprox(a_real,Pars.a,atol=0.2 )
    @test isapprox(n_real,Pars.n,atol=0.2 )
   
    REAL=[pa_real0,pa_real15,pa_real5,i_real0,i_real15,i_real5,v0_real,a_real,n_real]

    @show REAL
    @show logposterior(REAL)

    GalaxyWarp.update_parameters!(Pars,REAL)
    disks_real = GalaxyWarp.make_disks(Pars.Rd,Pars)
	GalaxyWarp.cloud_geometry!(clouds_sim,disks_real,Pars)
    fsky=GalaxyWarp.plot_sky(clouds_sim,disks,msize=7)
	save("$(@__DIR__)/sky_real.png",fsky)

	#GalaxyWarp.cloud_geometry!(clouds_sim,disks,Pars)
    
	fdisks=GalaxyWarp.plot_disks(Dict("MAP"=>disks,"d0"=>disks0,"REAL"=>disks_real))
	save("$(@__DIR__)/disks.png",fdisks)

    for ang in [0.0,15.0,30.0]
		pvdf=GalaxyWarp.plot_pvd(simdata,ang;disks=Dict("MAP"=>disks,"d0"=>disks0,"REAL"=>disks_real),clouds=clouds_sim,merged=true)
		save("$(@__DIR__)/pvd_$(ang).png",pvdf)
	end
end
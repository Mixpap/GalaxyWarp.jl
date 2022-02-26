using GalaxyWarp
using Test
using Parameters
using PCHIPInterpolation
using Optim
using BlackBoxOptim
using GLMakie 
using Distributions: Normal

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
    pixel_fit_steps=800
    kinematics_fittting_steps=5000

    simdata=GalaxyWarp.sim_data("$(@__DIR__)/test";theta_dist="uniform", Vcirc = x-> 320.0*x/(x^2.0+1.5^2.0)^1.2,I=x->70.0,PA=x->6.0*x, Sb = x->100.0, sigma_inst=x->10.0, cloud_mass=1e5 ,xl=3.0,yl=5.0,vl=350.0,dx=0.05,dy=0.05,dv=20.0,beam=[0.08,0.07],Rmin=0.1,dR=0.1,Rmax = 4.5,a=0.01,rms=0.05,Rcmax=2.1,Rcmap=1.7,k=3)

    @test length(simdata["Xc"])>100000
    @test length(simdata["Yc"])==length(simdata["Xc"])
    @test length(simdata["Vc"])==length(simdata["Xc"])
    @test length(simdata["Zc"])==length(simdata["Xc"])

    clouds_sim=GalaxyWarp.fit_pixelst(simdata;x0=-5.0,x1=5.0,y0=-5.5,y1=5.5,sigma=3.0, ncrit=1,N0=1,Nmax=7,vmin=-400.0, vmax=400.0,svmin=15.0, svmax=32.0,maxsteps=pixel_fit_steps, popsize=16)

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

    lenX1m=length(clouds_sim.Xc)
    @test lenX1m>10
    @test lenX1m<lenX1
    @test length(clouds_sim.Vc)==lenX1m
    @test length(clouds_sim.Yc)==lenX1m

    clouds_sim_f2=GalaxyWarp.filter_clouds(clouds_sim,[Dict("x0"=>-1.0)])
    lenX1_2=length(clouds_sim_f2.Xp)
    @test lenX1_2 == lenX1
    @test length(clouds_sim_f2.Vp)==lenX1_2
    @test length(clouds_sim_f2.Yp)==lenX1_2

    lenX1_2c=length(clouds_sim_f2.Xc)
    @test lenX1_2c < length(clouds_sim.Xc)
    @test length(clouds_sim_f2.Vc)==lenX1_2c
    @test length(clouds_sim_f2.Yc)==lenX1_2c 

    GalaxyWarp.cloud_fitting_diagnostics(simdata,clouds_sim_f1,"$(@__DIR__)/test")

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

    opt_L = bbsetup(logposterior;
		SearchRange = collect(zip(plower, pupper)),
		PopulationSize=120,MaxSteps = kinematics_fittting_steps
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

    ## Filtering by dV
    clouds_sim_res=GalaxyWarp.filter_clouds(clouds_sim,[Dict("dv"=>10.0)])
    GalaxyWarp.cloud_fitting_diagnostics(simdata,clouds_sim_res,"$(@__DIR__)/test_res")

    lenX1_3=length(clouds_sim_res.Xp)
    @test lenX1_3 < length(clouds_sim.Xp)
    @test length(clouds_sim_res.Vp)==lenX1_3
    @test length(clouds_sim_res.Yp)==lenX1_3 

    ## Checking how good is the fit
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


@testset "Noisy Simulated data Fitting" begin
    simdata=GalaxyWarp.sim_data("noisy_cube";theta_dist="uniform", Vcirc = x-> 320.0*x/(x^2.0+1.5^2.0)^1.2,I=x->70.0,PA=x->6.0*x, Sb = x->100.0, sigma_inst=x->10.0, cloud_mass=1e5 ,xl=3.0,yl=5.0,vl=350.0,dx=0.05,dy=0.05,dv=20.0,beam=[0.08,0.07],Rmin=0.1,dR=0.1,Rmax = 4.5,a=0.01,rms=0.05,Rcmax=2.1,Rcmap=1.7,k=3)

    simdata["data"] .=rand.(Normal.(simdata["data"], abs.(0.1 .* simdata["data"])))

    @show size(simdata["data"])
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

const G = 4.304574992e-06

@with_kw mutable struct p_mrk79_
    Fit::Array{Symbol}=[ 	
        :pa0,:pa15,:pa5,
        :i0,:i15,:i5,
        :v0,:a,:n]
    
    pa0::Float64=130.0
    pa15::Float64=130.0
    pa5::Float64=130.0
    
    PA::Array{Symbol}=[:pa0,:pa15,:pa5]
    R_pa::Vector{Float64}=[0.0,5.0,15.0]
    ppa = Interpolator
    
    i0::Float64=50.0
    i15::Float64=50.0
    i5::Float64=50.0
    
    I::Array{Symbol}=[:i0,:i15,:i5]
    R_i::Vector{Float64}=[0.0,5.0,15.0]
    ppi = Interpolator
    
    Mbh=1.0e8
    v0::Float64=300.0
    a::Float64= 2.0
    n::Float64= 1.0
    
    drc=0.1
    Rd=collect(0.01:drc:15.0)
    dR =0.1
    dV0=300.0
    dv=32.0
    V::Array{Symbol}=[:v0,:a,:n]

    Vcirc::Function = (r::Float64,v0::Float64,a::Float64,n::Float64) -> sqrt(
		G * Mbh / r + (v0*r/(r^2.0+a^2.0)^n)^2.0)
    
end

pwd()
#@testset "real data Fitting" begin
    data=GalaxyWarp.load_cube("$(@__DIR__)/data/COcube.fits", "CO21"; z=0.022296, rms=0.0012, centerx="median", centery="median",zaxis="vel", line_rest=230.538e9, x0=-15.5, x1=15.5, y0=-15.5, y1=15.5, v0=-550.0, v1=550.0)

    figmom21=GalaxyWarp.moments(data,1400,800;x0=-10.0,x1=10.0,y0=-10.0,y1=10.0,v0=-300.0,v1=300.0,dx=1.0,dv=5.0,vmax=80.0,denoise=4.5)


    #@show size(data["data"])
    @test length(data["Xc"])>100000
    @test length(data["Yc"])==length(data["Xc"])
    @test length(data["Vc"])==length(data["Xc"])
    @test length(data["Zc"])==length(data["Xc"])

    clouds_mrk79=GalaxyWarp.fit_pixelst(data;x0=-10.0,x1=10.0,y0=-10.5,y1=10.5,sigma=3.0, ncrit=1,N0=1,Nmax=7,vmin=-400.0, vmax=400.0,svmin=15.0, svmax=32.0,maxsteps=800, popsize=16)

    GalaxyWarp.cloud_fitting_diagnostics(data,clouds_mrk79,"$(@__DIR__)/mrk79")

    lenX0=length(clouds_mrk79.Xp)
    @test lenX0>2000
    @test length(clouds_mrk79.Yp)==lenX0
    @test length(clouds_mrk79.Vp)==lenX0
    @test length(clouds_mrk79.belongs)==0
    @test length(clouds_mrk79.belongs)==0

  
    data["beam"]
	pixels_per_beam=Int(round(data["beam"][1]*data["beam"][2]/data["dx"]^2.0))

    @time GalaxyWarp.merge_pixel_clouds!(clouds_mrk79;dr=1.5,dv=50.0,mp=10,df=3.0*data["rms"],roll_d=8,max_d=1.0)

    GalaxyWarp.cloud_fitting_diagnostics(data,clouds_mrk79,"$(@__DIR__)/mrk79_m";slit=1.0)


    lenX1m=length(clouds_mrk79.Xc)
    @test length(clouds_mrk79.Vc)==lenX1m
    @test length(clouds_mrk79.Yc)==lenX1m

    Pars=p_mrk79_()
    par0=copy([getfield(Pars,s) for s in Pars.Fit])
    GalaxyWarp.update_parameters!(Pars,par0)
    disks0 = GalaxyWarp.make_disks(Pars.Rd,Pars)

    function logposterior(pars::Vector{Float64})::Float64
		GalaxyWarp.update_parameters!(Pars,pars)
		likedV=GalaxyWarp.likelihood(clouds_mrk79,Pars;dv=30.0)
		#prior=GalaxyWarp.logprior(Pars,Priors)
		return likedV
	end

    #res =optimize(logposterior, par0, NelderMead(),Optim.Options(iterations = 3600, store_trace = true, show_trace = true,g_tol=1.0e-3))

	#MAP=Optim.minimizer(res)

    plower=[0.0,0.0,0.0,
    0.0,0.0,0.0,
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
	GalaxyWarp.cloud_geometry!(clouds_mrk79,disks,Pars)

    fsky=GalaxyWarp.plot_sky(clouds_mrk79,disks;msize=10,zoom_image=[-10.0,10.0,-10.0,10.0,2.0,2.0])
	save("$(@__DIR__)/sky.png",fsky)

    # pa_real0=0.0
    # pa_real15=6.0*1.5
    # pa_real5=6.0*5.0

    # i_real0=70.0
    # i_real15=70.0
    # i_real5=70.0

    # v0_real=320.0
    # a_real=1.5
    # n_real=1.2

    # dpa=7.0
    # di=7.0
    # @test isapprox(pa_real0,Pars.pa0,atol=dpa)
    # @test isapprox(pa_real15,Pars.pa15,atol=dpa)
    # @test isapprox(pa_real5,Pars.pa5,atol=dpa)
    # @test isapprox(i_real0,Pars.i0,atol=di)
    # @test isapprox(i_real15,Pars.i15,atol=di)
    # @test isapprox(i_real5,Pars.i5,atol=di)
    # @test isapprox(v0_real,Pars.v0,atol=20.0 )
    # @test isapprox(a_real,Pars.a,atol=0.2 )
    # @test isapprox(n_real,Pars.n,atol=0.2 )
   
    # REAL=[pa_real0,pa_real15,pa_real5,i_real0,i_real15,i_real5,v0_real,a_real,n_real]

    # @show REAL
    # @show logposterior(REAL)

    # GalaxyWarp.update_parameters!(Pars,REAL)
    # disks_real = GalaxyWarp.make_disks(Pars.Rd,Pars)
	# GalaxyWarp.cloud_geometry!(clouds_sim,disks_real,Pars)
    # fsky=GalaxyWarp.plot_sky(clouds_sim,disks,msize=7)
	# save("$(@__DIR__)/sky_real.png",fsky)

	#GalaxyWarp.cloud_geometry!(clouds_sim,disks,Pars)
    
	fdisks=GalaxyWarp.plot_disks(Dict("MAP"=>disks,"d0"=>disks0))
	save("$(@__DIR__)/mrk_79_disks.png",fdisks)

    for ang in [130.0]
		pvdf=GalaxyWarp.plot_pvd(data,ang;disks=Dict("MAP"=>disks,"d0"=>disks0),clouds=clouds_mrk79,merged=true,slit=1.0)
		save("$(@__DIR__)/mrk79_pvd_$(ang).png",pvdf)
	end
#end


const G = 4.304574992e-06
const K1 =0.0013329419990568908
function VNFW(R::Float64, M_vir::Float64, c::Float64)
	K2=119.73662477388707 * R * c
	return K1 *sqrt(M_vir) *
		sqrt((log(1.0 + K2 / M_vir^0.333333333) - K2 / (M_vir^0.3333333 *(1.0 + K2 / M_vir^0.333333333))
		) / (-c / (c + 1.0) + log(c + 1.0))) / sqrt(R)
end

@with_kw mutable struct p_5055_
    Rmax=7.5
    Fit::Array{Symbol}=[ 	
        :pa0,:pa15,:pa5,
        :i0,:i15,:i5,
        :v0,:a,:n]
    
    pa0::Float64=130.0
    pa15::Float64=130.0
    pa5::Float64=130.0
    
    PA::Array{Symbol}=[:pa0,:pa15,:pa5]
    R_pa::Vector{Float64}=[0.0,Rmax/2,Rmax]
    ppa = Interpolator
    
    i0::Float64=50.0
    i15::Float64=50.0
    i5::Float64=50.0
    
    I::Array{Symbol}=[:i0,:i15,:i5]
    R_i::Vector{Float64}=[0.0,Rmax/2,Rmax]
    ppi = Interpolator
    
    Mbh=1.0e8
    v0::Float64=300.0
    a::Float64= 2.0
    n::Float64= 1.0
    
    drc=0.2
    Rd=collect(0.01:drc:Rmax)
    dR =0.1
    dV0=300.0
    dv=32.0
    V::Array{Symbol}=[:v0,:a,:n]

    Vcirc::Function = (r::Float64,v0::Float64,a::Float64,n::Float64) -> sqrt(
		G * Mbh / r + (v0*r/(r^2.0+a^2.0)^n)^2.0)
    
end

using GalaxyWarp
pwd()
#@testset "real data Fitting" begin
    run(`mkdir -p test/NGC5055/`)
    data_5055_HI=GalaxyWarp.load_cube("$(@__DIR__)/data/NGC5055_HI_lab.fits", "HI"; z=0.001678, rms=0.00035, centerx="median", centery="median",beam=[0.1,0.1,0.0],zaxis="vel", x0=-15.5, x1=15.5, y0=-15.5, y1=15.5, v0=-550.0, v1=550.0)

    figmom=GalaxyWarp.moments(data_5055_HI,1400,800;x0=-10.0,x1=10.0,y0=-10.0,y1=10.0,v0=-300.0,v1=300.0,dx=1.0,dv=5.0,vmax=80.0,denoise=4.5)

    clouds_5055_HI=GalaxyWarp.fit_pixelst(data_5055_HI;x0=-10.0,x1=10.0,y0=-10.5,y1=10.5,sigma=3.0, ncrit=1,N0=1,Nmax=7,vmin=-400.0, vmax=400.0,svmin=15.0, svmax=32.0,maxsteps=800, popsize=16)

    ## CO21

    data_5055_CO=GalaxyWarp.load_cube("$(@__DIR__)/data/NGC5055_CO21_lab.fits", "CO21"; z=0.001678, rms=0.1, centerx="median", centery="median",zaxis="vel", x0=-7.5, x1=7.5, y0=-5.5, y1=5.5, v0=-550.0, v1=550.0) #beam=[0.1,0.1,0.0],)


    figmom=GalaxyWarp.moments(data_5055_CO,1400,800;x0=-7.0,x1=7.0,y0=-5.0,y1=5.0,v0=-300.0,v1=300.0,dx=1.0,dv=5.0,vmax=80.0,denoise=4.0)
    save("$(@__DIR__)/NGC5055/CO_moments.png",figmom)
    
    clouds_5055_CO=GalaxyWarp.fit_pixelst(data_5055_CO;x0=-7.0,x1=7.0,y0=-5.0,y1=5.0,sigma=3.0, ncrit=1,N0=1,Nmax=5,vmin=-340.0, vmax=340.0,svmin=3.0, svmax=55.0,maxsteps=1000, popsize=16)

    GalaxyWarp.cloud_fitting_diagnostics(data_5055_CO,clouds_5055_CO,"$(@__DIR__)/NGC5055/NGC5055")

    clouds_5055_CO_f=GalaxyWarp.filter_clouds(clouds_5055_CO,[Dict("f0"=>3.5)])

    GalaxyWarp.cloud_fitting_diagnostics(data_5055_CO,clouds_5055_CO_f,"$(@__DIR__)/NGC5055/NGC5055_filt")
  
    data_5055_CO["beam"]
	pixels_per_beam=Int(round(data_5055_CO["beam"][1]*data_5055_CO["beam"][2]/data_5055_CO["dx"]^2.0))

    @time GalaxyWarp.merge_pixel_clouds!(clouds_5055_CO;dr=0.5,dv=50.0,mp=10,df=3.0*data_5055_CO["rms"],roll_d=8,max_d=1.0)

    GalaxyWarp.cloud_fitting_diagnostics(data_5055_CO,clouds_5055_CO,"$(@__DIR__)/NGC5055/NGC5055_m";slit=0.1)


    lenX1m=length(clouds_5055_CO.Xc)
    @test length(clouds_5055_CO.Vc)==lenX1m
    @test length(clouds_5055_CO.Yc)==lenX1m

    Pars=p_5055_()
    par0=copy([getfield(Pars,s) for s in Pars.Fit])
    GalaxyWarp.update_parameters!(Pars,par0)
    disks0 = GalaxyWarp.make_disks(Pars.Rd,Pars)


    
    function logposterior(pars::Vector{Float64})::Float64
		GalaxyWarp.update_parameters!(Pars,pars)
		likedV=GalaxyWarp.likelihood(clouds_5055_CO,Pars;dv=30.0)
		#prior=GalaxyWarp.logprior(Pars,Priors)
		return likedV
	end

    #res =optimize(logposterior, par0, NelderMead(),Optim.Options(iterations = 3600, store_trace = true, show_trace = true,g_tol=1.0e-3))

	#MAP=Optim.minimizer(res)

    plower=[0.0,0.0,0.0,
    0.0,0.0,0.0,
    50.0,0.1,0.3]

    pupper=[360.0,360.0,360.0,
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
	GalaxyWarp.cloud_geometry!(clouds_5055_CO,disks,Pars)

    fsky=GalaxyWarp.plot_sky(clouds_5055_CO,disks;msize=10,zoom_image=[-10.0,10.0,-10.0,10.0,2.0,2.0])
	save("$(@__DIR__)/sky.png",fsky)

    # pa_real0=0.0
    # pa_real15=6.0*1.5
    # pa_real5=6.0*5.0

    # i_real0=70.0
    # i_real15=70.0
    # i_real5=70.0

    # v0_real=320.0
    # a_real=1.5
    # n_real=1.2

    # dpa=7.0
    # di=7.0
    # @test isapprox(pa_real0,Pars.pa0,atol=dpa)
    # @test isapprox(pa_real15,Pars.pa15,atol=dpa)
    # @test isapprox(pa_real5,Pars.pa5,atol=dpa)
    # @test isapprox(i_real0,Pars.i0,atol=di)
    # @test isapprox(i_real15,Pars.i15,atol=di)
    # @test isapprox(i_real5,Pars.i5,atol=di)
    # @test isapprox(v0_real,Pars.v0,atol=20.0 )
    # @test isapprox(a_real,Pars.a,atol=0.2 )
    # @test isapprox(n_real,Pars.n,atol=0.2 )
   
    # REAL=[pa_real0,pa_real15,pa_real5,i_real0,i_real15,i_real5,v0_real,a_real,n_real]

    # @show REAL
    # @show logposterior(REAL)

    # GalaxyWarp.update_parameters!(Pars,REAL)
    # disks_real = GalaxyWarp.make_disks(Pars.Rd,Pars)
	# GalaxyWarp.cloud_geometry!(clouds_sim,disks_real,Pars)
    # fsky=GalaxyWarp.plot_sky(clouds_sim,disks,msize=7)
	# save("$(@__DIR__)/sky_real.png",fsky)

	#GalaxyWarp.cloud_geometry!(clouds_sim,disks,Pars)
    
	fdisks=GalaxyWarp.plot_disks(Dict("MAP"=>disks,"d0"=>disks0))
	save("$(@__DIR__)/mrk_79_disks.png",fdisks)

    for ang in [0.0,70.0,130.0]
		pvdf=GalaxyWarp.plot_pvd(data_5055_CO,ang;disks=Dict("MAP"=>disks,"d0"=>disks0),clouds=clouds_5055_CO,merged=true,slit=0.2)
		save("$(@__DIR__)/NGC5055/NGC5055_pvd_$(ang).png",pvdf)
	end
#end
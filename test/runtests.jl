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

#@testset "Simulated data Fitting" begin
simdata=GalaxyWarp.sim_data("$(@__DIR__)/test";theta_dist="uniform", Vcirc = x-> 320.0*x/(x^2.0+1.5^2.0)^1.2,I=x->70.0,PA=x->6.0*x, Sb = x->100.0, sigma_inst=x->10.0, cloud_mass=1e5 ,xl=2.0,yl=5.0,vl=350.0,dx=0.05,dy=0.05,dv=20.0,beam=[0.08,0.07],Rmin=0.1,dR=0.1,Rmax = 4.5,a=0.01,rms=0.05,Rcmax=2.1,Rcmap=1.7,k=3,plots=true)

@test length(simdata["Xc"])>100000
@test length(simdata["Yc"])==length(simdata["Xc"])
@test length(simdata["Vc"])==length(simdata["Xc"])
@test length(simdata["Zc"])==length(simdata["Xc"])

clouds_sim=GalaxyWarp.fit_pixelst(simdata;x0=-5.0,x1=5.0,y0=-5.5,y1=5.5,sigma=3.0, ncrit=1,N0=1,Nmax=7,vmin=-400.0, vmax=400.0,svmin=15.0, svmax=32.0,maxsteps=800, popsize=16,debug=true)

GalaxyWarp.cloud_fitting_diagnostics(simdata,clouds_sim,"$(@__DIR__)/test")

@test length(clouds_sim.Xp)>2000

#end
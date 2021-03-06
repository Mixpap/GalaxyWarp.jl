@with_kw mutable struct Clouds
    Xp::Vector{Float64}
    Yp::Vector{Float64}
    Vp::Vector{Float64}
    Fp::Vector{Float64}
    Sp::Vector{Float64}
    Ip::Vector{Float64}
    logs::Dict{String, <:Any}

    belongs::Vector{Int64}=[]
    I::Vector{Int64}=[]
    k::Vector{Int64}=[]

    Xc::Vector{Float64}=[]
    Yc::Vector{Float64}=[]
    Vc::Vector{Float64}=[]
    Fc::Vector{Float64}=[]
    Sc::Vector{Float64}=[]
    Ic::Vector{Float64}=[]

    r::Vector{Float64}=[]
    R::Vector{Float64}=[]
    Φ::Vector{Float64}=[]
    VC::Vector{Float64}=[]
    dV::Vector{Float64}=[]
    Z::Vector{Float64}=[]
    inc::Vector{Float64}=[]
    pa::Vector{Float64}=[]
    dT::Vector{Float64}=[]
    Tnet::Vector{Float64}=[]
    ϕ::Vector{Float64}=[]
    tilt::Vector{Float64}=[]
end

""" 
chunk3(arr::Vector{Float64})

Splits a vector in chunks of size 3
## Examples
```jldoctest
    julia> chunk3([1.,2.,3.,4.,5.,6.])
    2-element Vector{Vector{Float64}}:
    [1.0, 2.0, 3.0]
    [4.0, 5.0, 6.0]
```
```jldoctest
    julia> chunk3([1.,2.,3.,4.,5.])
    2-element Vector{Vector{Float64}}:
    [1.0, 2.0, 3.0]
    [4.0, 5.0]
```
"""
function chunk3(arr::Vector{Float64})::Vector{Vector{Float64}}
    return [arr[i:min(i + 2, end)] for i in 1:3:length(arr)]
end

""" 
gaus(v,vc,A,sv) = ``A\\\\exp(\\\\frac{-(v-v_c)^2}{2 s_v^2.0})```

Gaussian function with scale ``A`` center at ``v_c`` and dispersion ``s_v``
## Examples
```jldoctest
    julia> gaus(80.0,100.0,0.1,37.0)
    0.08640781715118678
```
"""
function gaus(v::Float64,vc::Float64,A::Float64,sv::Float64)::Float64
    return A *exp(-0.5*(v-vc)^2.0 /sv^2.0)
end

""" 
gmodel!(s::Vector{Float64},v::Vector{Float64},p::Vector{Float64})::Vector{Float64}

Multiple gaussians one dimension model with coordinates ``v``.

The parameters of the gaussians are infered from splitted ``p`` to N*[vc,A,sv] where ``N`` is the number of gaussians (infered also from p)

## Examples
```jldoctest
julia>  V = collect(-350.0:20:350)
        s=gmodel(V,[0.1,120.0,23.0,0.12,-60.0,33.0])
    36-element Vector{Float64}: ....
```
"""
function gmodel(v::Vector{Float64},p::Vector{Float64})::Vector{Float64}
    s=zeros(size(v))
    pars=chunk3(p)
    for (A,vc,sv) in pars
        si = [gaus(vi,vc,A,sv) for vi in v]
        s = s .+ si
    end
    return s
end

""" 
gmodel_residual!(s::Vector{Float64},v::Vector{Float64},p::Vector{Float64})::Vector{Float64}

Sum of squared Residual of vector ``s`` after the subtraction of multiple gaussians one dimensional model with coordinates ``v``.

The parameters of the gaussians are infered from splitted ``p`` to N*[vc,A,sv] where ``N`` is the number of gaussians (infered also from p)

## Examples
```jldoctest
julia>  V = collect(-350.0:20:350)
        s=gmodel(V,[0.1,120.0,23.0,0.12,-60.0,33.0])
        gmodel_residual!(s,V,[0.1,120.0,23.0,0.12,-60.0,33.0])
        isapprox(gmodel_residual!(s,V,[0.1,120.0,23.0,0.12,-60.0,33.0]), 0.0; atol=eps(Float64), rtol=0)
        true
```
"""
function gmodel_residual!(s::Vector{Float64},v::Vector{Float64},p::Vector{Float64})::Float64
    pars=chunk3(p)
    for (A,vc,sv) in pars
        si = [gaus(vi,vc,A,sv) for vi in v]
        s = s .- si
    end
    return sum(s .^2.0)
end


""" 
fit_pixel(signal::Vector{Float},rms::Float64,vv::Vector{Float64},er::Float64;vmin=-400.0,vmax=400.0,svmin=15.0,svmax=30.0,smax=1000.0,maxsteps=100000,popsize=80,debug=false,N0=1,Nmax=8)

Fit a multiple gaussian model in ``signal`` by minimizing the Bayesian Information Criterion (BIC):
``\\mathrm{BIC}=k\\log(n)-2\\log(\\hat{L})`` where ``k3\\times N`` is the number of the parametes, ``n`` the signal size and ``\\hat{L}`` the maximum likelihood estimation for a ``N``-gaussians model. 
The algorithm fits the signal with multiple gaussians, calculating 

## Input
- ``signal::Vector{Float}`` The input signal
- ``rms::Float64`` The noise of the data (it used in scale boundaries).
- ``er::Float64`` The error of the data as it used in the likelihood.

## Parameters
### Model Parameters boundaries
- ``vmin`` The lower boundary on central velocities (-400)
- ``vmax`` The upper boundary on central velocities (400)
- ``svmin`` The lower boundary on velocity dispersion (15)
- ``svmax`` The upper boundary on velocity dispersion (30)
- ``smax`` The upper boundary on the scaling in units of rms (1000)
- ``smax`` The upper boundary on the scaling in units of rms (1000)
### Fitter Parameters
- ``maxsteps`` The steps of the fitter for every model fitting (30000)
- ``popsize`` Population size for the differential evolution global fitter
- ``N0`` The starting number of gaussians (1)
- ``Nmax`` Maximum number of gaussians (8)
- ``debug`` Make plots of the procedure (false)

## Output
Best Parameters, Minimum BIC

## Examples
```jldoctest
julia>  V = collect(-350.0:20:350)
        s=gmodel(V,[0.1,120.0,23.0,0.12,-60.0,33.0])
        bpars,minbic=fit_pixel(s,0.02,V,0.02;vmin=-400.0,vmax=400.0,svmin=15.0,svmax=30.0,smax=1000.0,maxsteps=20000,popsize=80,N0=1,Nmax=8)
    ([0.1255719888895172, -60.00001027466198, 30.0, 0.0999767960308078, 119.99487202319484, 23.012779807985947], -193.52458049526706)
```
## Notes
The likelihood function is: 
    ``2\\log\\hat{L} = -\\Big(\\sum_{i=1}^{n}\\frac{\\big(F_i-S(V,\\mathbf{V}_c,\\mathbf{F},\\mathbf{\\sigma})\\big)^2}{\\sigma_F^2} +n\\log(2\\pi\\sigma_F^2)\\Big)=-\\Big(\\chi^2+n\\log(2\\pi\\sigma_F^2)\\Big)``
"""
function fit_pixel(signal::Vector{Float64},rms::Float64,vv::Vector{Float64},er::Float64;vmin=-400.0,vmax=400.0,svmin=15.0,smin=0.75,svmax=30.0,smax=1000.0,maxsteps=30000,popsize=40,N0=1,Nmax=8)
    lowerc = [smin*rms,vmin,svmin]
    upperc = [smax*rms,vmax,svmax]

    lower=Float64[]
    upper=Float64[]

    minbic=1e10
    minchi=1000.0
    bic=minbic-11
    N=N0
    if N>1
        for ni in 1:N-1 #populate the boundaries
            lower=vcat(lower,lowerc)
            upper=vcat(upper,upperc)
        end
    end
    n=length(signal)
    bpars=Float64[]

    #chi_ = [minchi]
    L(p) = gmodel_residual!(signal,vv,p)
    while N<Nmax
        #@show N
        lower=vcat(lower,lowerc)
        upper=vcat(upper,upperc)
        k=length(lower)
        opt = bbsetup(L; SearchRange = collect(zip(lower, upper)),PopulationSize=popsize,MaxSteps = maxsteps, TraceMode = :silent)
        res1 = bboptimize(opt)
        pbb=best_candidate(res1)
        chi=best_fitness(res1)
        #@show pbb,chi
        logL=-0.5*(chi/er^2.0+n*log(2.0*pi*er^2.0))
       # @show logL
        bic=k*log(n)-2.0*logL
        #@show bic

        if (bic>minbic)
            break
        else
            minbic=bic
            bpars=pbb
            N+=1
        end
    end
    
    return bpars,minbic
end

""" 
fit_pixels(ipixels::Vector{CartesianIndex{2}},data::Dict{String, Any};x0=-2.0,x1=2.0,y0=-5.0,y1=5.0,sigma=3.0,sigmac=3.0,crit=3,N0=1,vmin=-400.0,vmax=400.0,svmin=15.0,svmax=28.0,maxsteps=8000,popsize=22)

Fit a multiple gaussian model for every spaxel in ipixels ``signal`` by minimizing the Bayesian Information Criterion (BIC):
``\\mathrm{BIC}=k\\log(n)-2\\log(\\hat{L})`` where ``k3\\times N`` is the number of the parametes, ``n`` the signal size and ``\\hat{L}`` the maximum likelihood estimation for a ``N``-gaussians model. 
The algorithm fits the signal with multiple gaussians, calculating 

## Input
- ``Data`` Dictionary Data 
## Parameters
### Data space boundaries and masking
- ``x0``, ``x1``, ``y0``, ``y1`` x,y boundaries in kpc
- ``sigma`` sigma mask on 0th moment

### Model Parameters boundaries
- ``vmin`` The lower boundary on central velocities (-400)
- ``vmax`` The upper boundary on central velocities (400)
- ``svmin`` The lower boundary on velocity dispersion (15)
- ``svmax`` The upper boundary on velocity dispersion (30)
- ``smax`` The upper boundary on the scaling in units of rms (1000)
- ``smax`` The upper boundary on the scaling in units of rms (1000)
### Fitter Parameters
- ``maxsteps`` The steps of the fitter for every model fitting in units of number of parameter=3N. So, for the first gaussian it will run ``maxsteps``, for the second ``3X maxsteps`` etc (30000)
- ``popsize`` Population size for the differential evolution global fitter
- ``N0`` The starting number of gaussians (1)
- ``Nmax`` Maximum number of gaussians (8)
- ``debug`` Make plots of the procedure (false)

## Output
Best Parameters, Minimum BIC

## Examples
```jldoctest
julia>  V = collect(-350.0:20:350)
        s=gmodel(V,[0.1,120.0,23.0,0.12,-60.0,33.0])
        bpars,minbic=fit_pixel(s,0.02,V,0.02;vmin=-400.0,vmax=400.0,svmin=15.0,svmax=30.0,smax=1000.0,maxsteps=20000,popsize=80,N0=1,Nmax=8)
    ([0.1255719888895172, -60.00001027466198, 30.0, 0.0999767960308078, 119.99487202319484, 23.012779807985947], -193.52458049526706)
```
"""
function fit_pixelst(data::Dict{String, Any};x0=-2.0,x1=2.0,y0=-5.0,y1=5.0, sigma=4.0,ncrit=1,N0=1,Nmax=8,vmin=-400.0,vmax=400.0, svmin=15.0,svmax=28.0, maxsteps=3000,popsize=22)
    beam_area=data["beam"][1]*data["beam"][2]*pi/(4.0*log(2.0)) #kpc^2
    pixel_area=data["dx"]^2.0
    mom0_data = sum(data["data"] .> sigma*data["rms"], dims = 3)[:, :, 1];
    ipixels=findall(mom0_data .> ncrit)
    @info "Starting line fitter on $(size(ipixels)[1]) spaxels"
    fit_results=[]
    Threads.@threads for n in 1:length(ipixels)
        ix=ipixels[n][1]
        iy=ipixels[n][2]
        x=data["X"][ix]
        y=data["Y"][iy]
        if ((x > x0) & (x < x1)) & ((y > y0) & (y < y1))
            signal=data["data"][ix,iy,:]
            pbb,_=fit_pixel(signal,sigma*data["rms"],data["V"],data["rms"];vmin=vmin,vmax=vmax,svmin=svmin,svmax=svmax,maxsteps=maxsteps,popsize=popsize,N0=N0,Nmax=Nmax)
            for gp in chunk3(pbb)
                A,vc,sv = gp
                Ip=A .* sv .* sqrt(2.0*pi) .*(pixel_area/beam_area) #jy km/s
                push!(fit_results,[x,y,vc,A,sv,Ip])
            end
        end
    end
    @info "Found $(length(fit_results)) subpixel clouds, dont forget to save them."
    cdp=Dict{String, Any}("svmax"=>svmax,"svmin"=>svmin,"x0"=>x0,"x1"=>x1,"y0"=>y0,"y1"=>y1,"vmin"=>vmin,"vmax"=>vmax,"sigma"=>sigma,"ncrit"=>ncrit,"maxsteps"=>maxsteps,"popsize"=>popsize,"N0"=>N0,"Nmax"=>Nmax)

	Xp=fill(NaN,length(fit_results))
	Yp=fill(NaN,length(fit_results))
	Vp=fill(NaN,length(fit_results))
	Fp=fill(NaN,length(fit_results))
	Sp=fill(NaN,length(fit_results))
	Ip=fill(NaN,length(fit_results))
	n=0
	for i in 1:length(fit_results)
		if isassigned(fit_results,i)
			n+=1
		    Xp[n]=fit_results[i][1]
			Yp[n]=fit_results[i][2]
			Vp[n]=fit_results[i][3]
			Fp[n]=fit_results[i][4]
			Sp[n]=fit_results[i][5]
			Ip[n]=fit_results[i][6]
		end
	end
    
    return Clouds(Xp=Xp,Yp=Yp,Vp=Vp,Fp=Fp,Sp=Sp,Ip=Ip,logs=cdp)
end

# function merge_pixel_clouds(X,Y,V,A;dx=0.1,dy=0.1,dv=30.0)
#     belongs = zeros(Int,length(X))
#     for i in reverse(sortperm(A)) #[1] #debug
#         #@show i #debug
#         if belongs[i] == 0 #if it hasnt merged to another source
#             belongs[i]=i #merge it to itself
#             Xi=X[i]
#             Yi=Y[i]
#             Vi=V[i]
#             Fi=A[i]
#             #@show Xi,Yi,Vi,Fi #debug
#             # find the distance to all other sources
#             dist= (Xi .-X) .^2.0 ./dx^2.0 .+(Yi .-Y) .^2.0 ./dy^2.0 .+(Vi .-V).^2.0 ./dv^2.0 #.<1.0
            
#             # sort the distance
#             local_sources_ind=sortperm(dist)#[1:50]
#             local_max_flux=Fi
#             #@show local_sources_ind
#             n=0
#             for j in local_sources_ind[2:end] #
#                 #@show n,j #debug
#                 d= dist[j]
#                 if belongs[j] == 0 #if it hasnt merged to another source
#                     n+=1
#                     #@show d #debug
#                     flux = A[j]
#                     #@show flux debug
#                     if flux <= local_max_flux
#                         belongs[j]=i#belongs[i]
#                     end
#                 end
#                 if  d>1
#                     break
#                 end
#             end
#         end
#     end
#     return belongs
# end

movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]

# function merge_pixel_clouds(X::Vector{Float64},Y::Vector{Float64},V::Vector{Float64},F::Vector{Float64},sV::Vector{Float64};dr=0.1,dv=30.0,mp=4,df=5000.0,roll_d=0.1,max_d=2.0,plots=false)
function merge_pixel_clouds!(clouds::Clouds;dr=0.1,dv=30.0,mp=4,df=5000.0,roll_d=0.1,max_d=2.0)
   
    X=clouds.Xp
    Y=clouds.Yp
    V=clouds.Vp
    F=clouds.Fp
    Ip=clouds.Ip
    sV=clouds.Sp
    
    @info "Merging $(length(X)) subpixel clouds"
    belongs = zeros(Int,length(X))
    for (n,i) in enumerate(reverse(sortperm(F))) #debug
        #@show i #debug
        #@show n,i
        if belongs[i] == 0 #if it hasnt merged to another source
            belongs[i]=i #merge it to itself
            Xi=X[i]
            Yi=Y[i]
            Vi=V[i]
            Fi=F[i]
            dist= (Xi .-X) .^2.0 ./dr^2.0 .+(Yi .-Y) .^2.0 ./dr^2.0 .+(Vi .-V).^2.0 ./dv^2.0
            pd=sortperm(dist)
            d=dist[pd]
            Flux=F[pd]
            if sum(d .< max_d)>mp
                ma=movingaverage(Flux[d .< max_d],roll_d)
                # if n==plot
                #     fig,ax,sc=lines(d[d .< max_d],Flux[d .< max_d])
                #     lines!(ax,d[d .< max_d],ma)
                #     #vlines!(ax,1.0)
                #     #display(fig)
                # end
                f0=Fi
                #f=f0
                
                for (maj,j) in zip(ma,pd[d .< max_d])
                    
                    if (belongs[j]==0)
                        f=maj
                        if (f<=f0+df) 
                            belongs[j]=i
                            f0=f
                            # if n==plot
                            #     scatter!(ax,[dist[d .< max_d][j]],[F[d .< max_d][j]],marker='X')
                            #     display(fig)
                            #     sleep(0.05)   
                            # end
                        else
                            break
                        end
                    end
                end
            end

        end
    end
    I_pre = unique(belongs)
    I=Vector{Int}()
    sI=Vector{Float64}()
    sIc=Vector{Float64}()
    for i in I_pre
        if sum(belongs .==i) > mp
            push!(I,i)
            push!(sI,sqrt(mean(sV[belongs .==i])^2.0 +std(V[belongs .==i] .- V[i])^2.0))
            push!(sIc,sum(Ip[belongs .==i]))
        end
    end
    @info "Found $(length(I)) unique clouds"
    
    setfield!(clouds, :belongs, belongs)
    setfield!(clouds, :I, I)
    setfield!(clouds, :Sc, sI)
    setfield!(clouds, :Xc, X[I])
    setfield!(clouds, :Yc, Y[I])
    setfield!(clouds, :Vc, V[I])
    setfield!(clouds, :Fc, F[I])
    setfield!(clouds, :Ic, sIc)
    setfield!(clouds, :logs, merge(clouds.logs,Dict{String, Any}("dr"=>dr,"dv"=>dv,"mp"=>mp,"df"=>df,"roll_d"=>roll_d,"max_d"=>max_d)))
end


function filter_clouds(clouds,rect::Vector{Dict{String, Float64}};neg=false,x0=-10.0,x1=10.0,y0=-10.0,z0=-10.0,z1=10.0,y1=10.0,v0=-1500.0,v1=1500.0,s0=0.0,s1=500.0,f0=0.0,f1=100.0,dv0=-1000.0,dv1=1000.0,R0=0.0,R1=100.0,r0=0.0,r1=100.0)

    mask=clouds.Xp .>1000.0
    maskc=clouds.Xc .>1000.0

	@info "Filtering $(sum(.!mask)) clouds"
    for rd in rect
		@info "Applying mask $(rd)"
        x0i = haskey(rd,"x0") ? rd["x0"] : x0
        x1i = haskey(rd,"x1") ? rd["x1"] : x1

        y0i = haskey(rd,"y0") ? rd["y0"] : y0
        y1i = haskey(rd,"y1") ? rd["y1"] : y1

        z0i = haskey(rd,"z0") ? rd["z0"] : z0
        z1i = haskey(rd,"z1") ? rd["z1"] : z1

		R0i = haskey(rd,"R0") ? rd["R0"] : R0
        R1i = haskey(rd,"R1") ? rd["R1"] : R1

		r0i = haskey(rd,"r0") ? rd["r0"] : r0
        r1i = haskey(rd,"r1") ? rd["r1"] : r1

        v0i = haskey(rd,"v0") ? rd["v0"] : v0
        v1i = haskey(rd,"v1") ? rd["v1"] : v1

		vc0i = haskey(rd,"vc0") ? rd["vc0"] : v0
        vc1i = haskey(rd,"vc1") ? rd["vc1"] : v1

        s0i = haskey(rd,"s0") ? rd["s0"] : s0
        s1i = haskey(rd,"s1") ? rd["s1"] : s1

        f0i = haskey(rd,"f0") ? rd["f0"] : f0
        f1i = haskey(rd,"f1") ? rd["f1"] : f1

		mask_rec= ((clouds.Xp .>x0i) .&& (clouds.Xp .<x1i)) .&& ((clouds.Fp .>f0i) .&& (clouds.Fp .<f1i)) .&& ((clouds.Yp .>y0i) .&& (clouds.Yp .<y1i)) .&& ((clouds.Vp .>v0i) .&& (clouds.Vp .<v1i)) .&& ((clouds.Sp .>s0i) .&& (clouds.Sp .<s1i)) 
		#@show sum(mask_rec)
		if length(clouds.r)>0
            dv0i = haskey(rd,"dv0") ? rd["dv0"] : dv0
			dv1i = haskey(rd,"dv1") ? rd["dv1"] : dv1
            mask_rec=mask_rec .&& ((clouds.dV .>dv0i) .&& (clouds.dV .<dv1i) )  .&& ((clouds.VC .>vc0i) .&& (clouds.VC .<vc1i)) .&& ((clouds.R .>R0i) .&& (clouds.R .<R1i)) .&& ((clouds.r .>r0i) .&& (clouds.r .<r1i)) .&& ((clouds.Z .>z0i) .&& (clouds.Z .<z1i))
			#@show sum(mask_rec)
        end

		@info "$(sum(mask_rec)) cloudlets in mask"
		mask =mask .|| (mask_rec)
		#@show sum(mask)

		# mask_rec= ((clouds.Xc .>x0i) .&& (clouds.Xc .<x1i)) .&& ((clouds.Fc .>f0i) .&& (clouds.Fc .<f1i)) .&& ((clouds.Yc .>y0i) .&& (clouds.Yc .<y1i)) .&& ((clouds.Vc .>v0i) .&& (clouds.Vc .<v1i)) .&& ((clouds.Sc .>s0i) .&& (clouds.Sc .<s1i)) 

		# maskc=maskc .|| (.!mask_rec)
    end
	if neg
		mask=.! mask
		@show sum(mask)
		maskc=.! maskc
	end
	
    @info "Found $(sum(mask)) unique sub-clouds and $(sum(maskc)) merged clouds"

    if length(clouds.r)>0 && length(clouds.ϕ)>0
        return GalaxyWarp.Clouds(Xp=clouds.Xp[mask],Yp=clouds.Yp[mask],Vp=clouds.Vp[mask],Fp=clouds.Fp[mask],Sp=clouds.Sp[mask],Ip=clouds.Ip[mask],logs=merge(clouds.logs,Dict{String, Any}("filtered"=>true)),belongs=clouds.belongs[mask],I=clouds.I[maskc],k=clouds.k[mask],Xc=clouds.Xc[maskc],Yc=clouds.Yc[maskc],Vc=clouds.Vc[maskc],Fc=clouds.Fc[maskc],Sc=clouds.Sc[maskc],Ic=clouds.Ic[maskc],r=clouds.r[mask],R=clouds.R[mask],Φ=clouds.Φ[mask],VC=clouds.VC[mask],dV=clouds.dV[mask],Z=clouds.Z[mask],inc=clouds.inc[mask],pa=clouds.pa[mask],dT=clouds.dT[mask],Tnet=clouds.Tnet[mask],ϕ=clouds.ϕ[mask],tilt=clouds.tilt[mask])
    elseif length(clouds.r)>0
        return GalaxyWarp.Clouds(Xp=clouds.Xp[mask],Yp=clouds.Yp[mask],Vp=clouds.Vp[mask],Fp=clouds.Fp[mask],Sp=clouds.Sp[mask],Ip=clouds.Ip[mask],logs=merge(clouds.logs,Dict{String, Any}("filtered"=>true)),belongs=clouds.belongs[mask],I=clouds.I[maskc],k=clouds.k[mask],Xc=clouds.Xc[maskc],Yc=clouds.Yc[maskc],Vc=clouds.Vc[maskc],Fc=clouds.Fc[maskc],Sc=clouds.Sc[maskc],Ic=clouds.Ic[maskc],r=clouds.r[mask],R=clouds.R[mask],Φ=clouds.Φ[mask],VC=clouds.VC[mask],dV=clouds.dV[mask],Z=clouds.Z[mask],inc=clouds.inc[mask],pa=clouds.pa[mask],dT=clouds.dT[mask],Tnet=clouds.Tnet[mask])
    else
        return GalaxyWarp.Clouds(Xp=clouds.Xp[mask],Yp=clouds.Yp[mask],Vp=clouds.Vp[mask],Fp=clouds.Fp[mask],Sp=clouds.Sp[mask],Ip=clouds.Ip[mask],logs=merge(clouds.logs,Dict{String, Any}("filtered"=>true)))
    end
end
#
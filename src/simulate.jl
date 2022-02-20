function _model_cube(data::Dict{String, Any}, IX::Vector{Int64}, XX::Vector{Float64},IY::Vector{Int64},YY::Vector{Float64},Sd::Function,Sdpars::Vector{Float64},Rd::Vector{Float64},dR::Float64,disks::Disks;search_v=1,beam=nothing,lowest_sig=3.0)::Array{Float64, 3}
	cube=copy(data["data"])
	model_cube=fill(0.0,size(cube))
	@progress for n in 1:length(IX)
		ix=IX[n]
		x=XX[n]
		iy=IY[n]
		y=YY[n]
		for (rd,i_d,phi_d,V_d) in zip(disks.R,disks.I,disks.PA,disks.V)
			d,xy_ellipse=distf(rd, i_d, phi_d, [x,y])
			if d<=dR
				phi_ell=atan(y,x)#xy_ellipse[2],xy_ellipse[1])
				Vd_sky=V_d*sin(i_d)*sin(phi_ell-phi_d)/sqrt(1.0 +tan(i_d)^2.0*cos(phi_ell-phi_d)^2.0)
				loc=abs.(Vd_sky.-data["V"]) .<search_v
				if any(loc)
					lsic=cube[ix,iy,:]
					local_sig=lsic[loc]
					Vlocal=data["V"][loc]
					Fdata=maximum(local_sig)
					Vmax=Vlocal[argmax(local_sig)]
					if Fdata > lowest_sig*data["rms"]
						msig=Fdata .*exp.(-(data["V"] .-Vmax) .^2.0 ./(2.0*Sd(rd,Sdpars...)^2.0))
						model_cube[ix,iy,:]+=msig
						cube[ix,iy,:] -=msig
					end
				end
			end
		end
	end
	if ~isnothing(beam)
		for iv in eachindex(data["V"])
			model_cube[:,:,iv]=imfilter(model_cube[:,:,iv],beam)
		end
	end
	return model_cube
end

function sim_data(name;theta_dist="uniform", Vcirc = x-> 100.0*sqrt(x),I= x->70.0,PA=x->33.0, Sb = x->100.0, sigma_inst=x->10.0, cloud_mass=1e5 ,xl=2.0,yl=5.0,vl=350.0,dx=0.05,dy=0.05,dv=20.0,beam=[0.08,0.07,32.0],Rmin=0.2,dR=0.1,Rmax = 4.5,a=1000.0,rms=0.05,Rcmax=2.1,Rcmap=1.7,k=3,plots=false)

    XX=[]
    YY=[]
    ZZ=[]
    VV=[]
    RR=[]
    Rc =[]
    M=[]
    sigma=[]

    for Ri in Rmin:dR:Rmax
        Nclouds = Int(round( 1.0e6*pi*((Ri+dR)^2.0 - (Ri-dR)^2.0) * Sb(Ri)/cloud_mass ))
        R=sqrt.(rand(Uniform((Ri-dR)^2.0,(Ri+dR)^2.0),Nclouds))

        if theta_dist == "uniform"
            theta=rand(Uniform(0.0,2.0*pi),Nclouds)
        else
            R0=0.1
            theta0=deg2rad(50.0)
            pitch=deg2rad(50.0)
            theta=log.(R ./R0) ./tan.(pitch) .+theta0
            append!(theta,log.(R ./ R0) ./tan.(pitch) .+(theta0+pi))
            append!(R,R)
        end

        Rci=10.0 .^rand(TriangularDist(1.0,Rcmax,Rcmap),Nclouds)
        Mi = Sb(Ri) .*Rci .^2
        sigmai = sqrt.( (0.7 .*sqrt.(Rci)).^2.0 .+(sigma_inst.(Ri)).^2.0)
        i=deg2rad(I(Ri))
        phi0=deg2rad(PA(Ri) .+90.0)
        phi=atan.(cos(i) *tan.(theta)) .+ phi0 
        r = R.*cos.(theta) ./cos.(phi .-phi0  )
        Xc=r .* cos.(phi)
        Yc=r .* sin.(phi)
        Zc = -tan.(i) .* (Xc .*cos.(deg2rad(PA(Ri))) .+Yc .*sin.(deg2rad(PA(Ri))))
        VLOS = Vcirc.(R) .* sin(i) .* cos.(theta)
        push!(XX,Xc)
        push!(YY,Yc)
        push!(ZZ,Zc)
        push!(VV,VLOS)
        push!(RR,R)
        push!(Rc,Rci)
        push!(M,Mi)
        push!(sigma,sigmai)
        # if plots
        #     Rd_sky=Ri ./(1.0 .+tan(deg2rad(I(Ri)))^2.0 .* cos.(phis .-deg2rad(PA(Ri))).^2.0).^0.5
        #     xd=Rd_sky .*cos.(phis)
        #     yd=Rd_sky .*sin.(phis)
        #     lines!(axm0,xd,yd,ratio=1,label=false,transparency = true,color=RGBA(0.0,0.0,0.0,0.1))
        # end
    end
    Xc=collect(Iterators.flatten(XX))
    Yc=collect(Iterators.flatten(YY))
    Zc=collect(Iterators.flatten(ZZ))
    Vc=collect(Iterators.flatten(VV))
    Rc=collect(Iterators.flatten(Rc))
    RR=collect(Iterators.flatten(RR))
    M=collect(Iterators.flatten(M))
    sigma=collect(Iterators.flatten(sigma))

    # if plots
        
    # end
    Y = collect(-yl:dx:yl)
    X = collect(-xl:dy:xl)
    V = collect(-vl:dv:vl)

    s=zeros(length(X),length(Y),length(V));
    Fc=[]
    for (r,xc,yc,vc,sx,sy,sv) in zip(RR,Xc,Yc,Vc,Rc ./1000,Rc ./1000,sigma)
        F=a *Sb(r)/sv
        push!(Fc,F)
        ix=argmin(abs.(xc.-X))
        iy=argmin(abs.(yc.-Y))
        iv=argmin(abs.(vc.-V))

        dix=max(Int(round(sx/dx)),1)
        diy=max(Int(round(sy/dx)),1)
        div=max(Int(round(sv/dv)),1)

        xr=ix-k*dix:ix+k*dix
        yr=iy-k*diy:iy+k*diy
        vr=iv-k*div:iv+k*div
        if (minimum(xr)>0) & (minimum(yr)>0) & (minimum(vr)>0) & (maximum(xr)<size(s)[1]) & (maximum(yr)<size(s)[2]) & (maximum(vr)<size(s)[3])
            s[xr,yr,vr]=s[xr,yr,vr].+ [F *exp(-0.5*( ((x-ix)/dix)^2.0 + ((y-iy)/diy)^2.0 +((v-iv)/div)^2.0)) for x in xr, y in yr, v in vr]
        end
    end
    s=rand.(Normal.(s,rms))
    bx = Int(round(beam[1]/dx))
    by = Int(round(beam[2]/dy))
    for i in 1:size(s)[3]
        imfilter!(s[:,:,i],s[:,:,i],Kernel.gaussian([bx,by]))
    end

    if plots
        figd=Figure(resolution = (1000,1500))
        axRc= Axis(figd[1,1],Aspect=:equal)
        axM= Axis(figd[1,2],Aspect=:equal)
        axS= Axis(figd[1,3],Aspect=:equal)
        
        scatter!(axRc,RR,Rc,markersize=1,trasnparency=true)
        #scatter!(axM,Xc,Yc,markersize=1,trasnparency=true)
        scatter!(axM,RR,log10.(M),markersize=1,trasnparency=true)
        scatter!(axS,RR,sigma,markersize=1,trasnparency=true)
        #scatter!(axxy,Xc,Yc,color=Vc,colorrange=[-300.0,300.0],colormap=:seismic)
        #display(fig)
        save("$(name)_data.png",figd)
        fig=Figure(resolution = (2200,1500))
        #axm0v = Axis(fig[1,1],Aspect=:equal)
        axm0 = Axis(fig[1,1],Aspect=:equal)
        axm1 = Axis(fig[1,2],Aspect=:equal)
        axm2 = Axis(fig[1,3],Aspect=:equal)
        #ax3d = Axis(fig[1,4],Aspect=:equal)
        axpvd1=Axis(fig[2,1],Aspect=:equal)
        axpvd2=Axis(fig[2,2],Aspect=:equal)
        phis=0.0:0.01:2*pi

        mom0=sum(s,dims=3)[:,:,1]
        mom1=[sum(V .* s[ix,iy,:]) for ix in 1:size(s,1), iy in 1:size(s,2)] ./mom0
        mom2=sqrt.(abs.([sum((V .- mom1[ix,iy]) .^2.0 .* s[ix,iy,:]) for ix in 1:size(s,1), iy in 1:size(s,2)] ./mom0))
        #heatmap!(axm0v,Y,V,sum(s,dims=2)[:,1,:])
        hm=heatmap!(axm0,X,Y,mom0)
        cb=Colorbar(fig[1, 1][1,2], hm)
        hm=heatmap!(axm1,X,Y,mom1,colorrange=[-300.0,300.0],colormap=:seismic)
        cb=Colorbar(fig[1, 2][1,2], hm)
        hm=heatmap!(axm2,X,Y,mom2,colorrange=[0.0,60.0])
        cb=Colorbar(fig[1, 3][1,2], hm)
        
        #scatter!(ax3d,Xc,Yc,Zc)

        pa=0.0
        θ=deg2rad(pa)
        k=3
        A=zeros(size(s))
        for i in 2:size(s)[3]
            A[:,:,i]=collect(imrotate(s[:,:,i], θ,size(s[:,:,1]),fill=0.0))[:,:,1]
        end
        #A[isnan.(A)] .=0.0
        pvA= sum(A[Int(floor(size(A)[1]/2))-k:Int(floor(size(A)[1]/2))+k,:,:], dims=1)
        contourf!(axpvd1,Y,V,pvA[1,:,:])

        pa=90.0
        θ=deg2rad(pa)
        k=3
        A=zeros(size(s))
        for i in 2:size(s)[3]
            A[:,:,i]=collect(imrotate(s[:,:,i], θ,size(s[:,:,1]),fill=0.0))[:,:,1]
        end
        #A[isnan.(A)] .=0.0
        pvA= sum(A[Int(floor(size(A)[1]/2))-k:Int(floor(size(A)[1]/2))+k,:,:], dims=1)
        contourf!(axpvd2,Y,V,pvA[1,:,:])
        save("$(name)_moments_pvd.png",fig)

        ff,ss,aa=scatter(Xc,Yc,Zc,axis = (; type = Axis3, protrusions = (0, 0, 0, 0),
        viewmode = :fit, limits = (-5, 5, -5, 5, -5, 5)))
        save("$(name)_3d.png",ff)
        #display(ff)
    end
    
    return Dict("name"=>name,"data"=>Float64.(s),"X"=>X,"Y"=>Y,"V"=>V,"dx"=>dx,"dv"=>dv,"Xc"=>Xc,"Yc"=>Yc,"Zc"=>Zc,"Vc"=>Vc,"Fc"=>Fc,"Sc"=>sigma,"rms"=>rms,"beam"=>beam)
end;
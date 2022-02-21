struct Disks
	R::Vector{Float64}
	PA::Vector{Float64}
	I::Vector{Float64}
	V::Vector{Float64}
end

# struct Geometry
# 	R::Vector{Float64}
# 	Rs::Vector{Float64}
# 	PA::Vector{Float64}
# 	I::Vector{Float64}
# 	M::Vector{Float64}
# 	X::Vector{Float64}
# 	Y::Vector{Float64}
# 	V::Vector{Float64}
# 	VC::Vector{Float64}
# 	dV::Vector{Float64}
# 	Z::Vector{Float64}
# 	Φ::Vector{Float64}
# 	dT::Vector{Float64}
# 	T::Vector{Float64}
# end

"""
Rotation Matrix
"""
function rot2d(angle::Float64)::Array{Float64,2}
    s, c = sin(angle), cos(angle)
    return [c -s ; s c]
end
#
"""
Minimum Distance between an ellipse and a point
algorithm from https://blog.chatfield.io/simple-method-for-distance-to-ellipse/

Input
======
- ``a`` major axis of the ellipse
- ``inclination`` inclination of the ellipse
- ``pa`` position angle of the ellipse
- ``p`` xy vector of the point

Parameters
=======
- ``debug`` plot 

Output
====
distance, xy vector of the closest point on the ellipse
"""
function distf(a::Float64, inclination::Float64, pa::Float64, p::Vector{Float64}; debug=false,tx=0.70710678118654752,ty=0.70710678118654752)
    angle=pa+pi/2.0
    p_rot = rot2d(-angle)*p
    p_abs=abs.(p_rot)
    
    b= abs(cos(inclination)*a)
    ss_sub = a^2.0 - b^2.0
    #local x=0.0 # =Float64()
    #y =Float64()
    for i in 1:3
        @fastmath x = tx*a
        @fastmath y = ty*b
        @fastmath eM = ss_sub*tx^3.0/a
        @fastmath em = -ss_sub*ty^3.0/b
        #eM = ss_sub*cos(tx)^3.0/semi_major
        #em = -ss_sub*cos(ty)^3.0/semi_minor
        rx=x-eM
        ry=y-em

        qx=p_abs[1]-eM
        qy=p_abs[2]-em

        r=hypot(rx,ry)
        q=hypot(qx,qy)

        # tx= clamp((qx*r/q+eM)/semi_major,0.0,1.0)
  #       ty= clamp((qy*r/q+em)/semi_minor,0.0,1.0)
        @fastmath tx= (qx*r/q+eM)/a
        @fastmath ty= (qy*r/q+em)/b
        tn=hypot(tx,ty)
        tx /= tn
        ty /= tn
        #@show x,y,tx,ty
    end
    x=a*tx
    y=b*ty

    x_edge=copysign(x, p_rot[1])
    y_edge=copysign(y, p_rot[2])
    p_edge = [x_edge,y_edge]
    dst = norm(p_edge .- p_rot, 2)
    p_edge_rot = rot2d(angle)*p_edge
    if debug
        fig=Figure()
        ax = Axis(fig[1, 1])#, xlabel = "x label", ylabel = "y label",title = "Title")
        ax.aspect = DataAspect()
        phis=0.0:0.01:2*pi
        R=a ./ sqrt.(1.0 .+ tan(inclination)^2.0 .* cos.(phis .-pa).^2.0)
        xd=R.*cos.(phis)
        yd=R .*sin.(phis)
        lines!(ax,xd,yd,ratio=1,label=false)
        GLMakie.scatter!(ax,[p[1]],[p[2]],label=false)
        #scatter!(pl,[p_rot[1]],[p_rot[2]],label="p_rot")
        GLMakie.scatter!(ax,[p_edge_rot[1]],[p_edge_rot[2]],label=false)
        lines!(ax,[p[1],p_edge_rot[1]],[p[2],p_edge_rot[2]])
        @show norm(p-p_edge_rot,2)
        @show dst, p_edge_rot
        display(fig)
    end
    return dst, p_edge_rot
end

function make_disks(Rd::Vector{Float64},P)::Disks
	
	PA = [getfield(P,s) for s in P.PA]
	I = [getfield(P,s) for s in P.I]
	V = [getfield(P,s) for s in P.V]

	PAd=deg2rad.(P.ppi(P.R_i,PA).(Rd))
	Id=deg2rad.(P.ppa(P.R_pa,I).(Rd))
	Vd=P.Vcirc.(Rd,V...)
	
	return Disks(Rd,PAd,Id,Vd)
end


function distrv(x::Float64,y::Float64,v::Float64,disks::Disks,dR::Float64;dvi_min=300.0)::Float64
	dVi=dvi_min
	for (rd,i_d,phi_d,V_d) in zip(disks.R,disks.I,disks.PA,disks.V)
		d,xy_ellipse=distf(rd, i_d, phi_d, [x,y])
		if d<=dR
			phi_ell=atan(y,x)
			Vd_sky= V_d* sin(i_d)*sin(phi_ell-phi_d)/sqrt(1.0 +tan(i_d)^2.0*cos(phi_ell-phi_d)^2.0)
			dvi=abs(Vd_sky-v)
			if dvi<dvi_min
				dvi_min=dvi
				dVi=Vd_sky-v
			end
		end
	end
	return dVi
end
#X::Vector{Float64}, Y::Vector{Float64}, V::Vector{Float64}
function cloud_find(disks::Disks,clouds::Clouds,P;dV0=300.0)::Vector{Float64}
	dV=fill(dV0,length(clouds.Xc))
    Threads.@threads for i in 1:length(clouds.Xc)
		x=clouds.Xc[i]
		y=clouds.Yc[i]
		v=clouds.Vc[i]
		@inbounds dV[i]=distrv(x,y,v,disks,P.dR;dvi_min=300.0)
    end
	return dV
end

function update_parameters!(P,pars::Vector{Float64})
	for (p,s) in zip(pars,P.Fit)
		setfield!(P,s,p)
	end
end


function likelihood(clouds::Clouds,P; dV0=300.0,dv=30.0,dv_hi=30.0):Float64
	dV=cloud_find(make_disks(P.Rd,P),clouds,P)
	return sum(dV .^2.0) /dv^2.0 	
end

function logprior(P,priors)::Float64
	ps=0.0
	for p in fieldnames(typeof(priors))
		pr=getfield(priors,p)
		ps += (getfield(P,p) - pr[1] )^2.0/ pr[2]^2.0
	end
	return ps
end

function cloud_properties(x::Float64,y::Float64,v::Float64,disks::Disks,dR::Float64;τ=nothing,dvi_min=300.0)::NTuple{7, Float64}
	dvc=dvi_min
	found=false
	zd=NaN
	ic=NaN
	vc=NaN
	phic=NaN
	dT=NaN
	Tnet=NaN
	for (rd,i_d,phi_d,V_d) in zip(disks.R,disks.I,disks.PA,disks.V)
		d,xy_ellipse=distf(rd, i_d, phi_d, [x,y])
		if d<=dR
			phi_ell=atan(y,x)
			Vd_sky= V_d* sin(i_d)*sin(phi_ell-phi_d)/sqrt(1.0 +tan(i_d)^2.0*cos(phi_ell-phi_d)^2.0)
			
			dvi=abs(Vd_sky-v)
			if dvi<dvi_min
				dvi_min=dvi
				dvc=Vd_sky-v
				vc=V_d
				zd = -tan(i_d)*(xy_ellipse[1]*cos(phi_d)+xy_ellipse[2]*sin(phi_d))
				ic=i_d
				phic=phi_d
				rc=rd
				found =true
			end
		end
	end
	if found
        if isnothing(τ)
            return dvc,vc,zd,ic,phic,NaN,NaN
        else
            Wd_ = [sin(ic)*cos(phic),sin(ic)*sin(phic),cos(ic)]
            Nd_= Wd_ ./norm(Wd_)
            Id_= [-sin(phic),cos(phic),0.0]
            Id_=Id_ ./norm(Id_)
            Md_=[cos(ic)*cos(phic),cos(ic)*sin(phic),-sin(ic)]#cross(Nd_,Id_)
            Md_=Md_ ./norm(Md_)
            
            
            dT=dot(τ([x,y,zd]),Nd_)
            τ_net(ϕ)=dot(τ(rc .*cos(ϕ) .* Id_ .+ rc .* sin(ϕ) .*Md_),Nd_)
            Tnet=rc*3.0857e16  /vc * quadgk(τ_net,0.0,2*pi;atol=0.1)[1]
            return dvc,vc,zd,ic,phic,dT,Tnet
        end
	else 
		return NaN,NaN,NaN,NaN,NaN,NaN,NaN
	end
end

function cloud_geometry!(clouds::Clouds,disks::Disks,P;dV0=300.0,τ=nothing)#::Geometry

    X=clouds.Xp
    Y=clouds.Yp
    V=clouds.Vp
    M=clouds.Ip
    
	VC=fill(NaN,length(X))
    Z=fill(NaN,length(X))
	K=fill(NaN,length(X))
	dT=fill(NaN,length(X))
	Tnet=fill(NaN,length(X))
    PA=fill(NaN,length(X))
    I=fill(NaN,length(X))
    FF=Bool.(zeros(length(X)))
	dV=fill(dV0,length(X))
    Threads.@threads for k in 1:length(X)
		x=X[k]
		y=Y[k]
		v=V[k]
        dvi_min=dV0
		
		κ1,κ2,κ3,κ4,κ5,κ6,κ7=cloud_properties(x,y,v,disks,P.dR;τ=τ,dvi_min=dV0)
		dV[k]=κ1
		VC[k]=κ2
		Z[k]=κ3
		I[k]=κ4
		PA[k]=κ5
		dT[k]=κ6
		Tnet[k]=κ7
    end

    clouds.r=sqrt.(X .^2.0 .+Y .^2.0 )
    clouds.R=sqrt.(X .^2.0 .+Y .^2.0 .+Z .^2.0)
    clouds.Φ=atan.(Y,X)
    clouds.Z=Z
    clouds.inc=I
    clouds.pa=PA
    clouds.dT=dT
    clouds.dV=dV
	#return Geometry(sqrt.(X .^2.0 .+Y .^2.0 .+Z .^2.0), sqrt.(X .^2.0 .+Y .^2.0 ),PA,I,M,X,Y,V,VC,dV,Z,atan.(Y,X),dT,Tnet)
end


function manipulate_cube(ix::Int64,x::Float64,iy::Int64,y::Float64,cube::Array{Float64, 3},model_cube::Array{Float64, 3},data::Dict{String, Any},disks::Disks,search_v::Float64,Sd::Function,Sdpars::Vector{Float64},Rd::Vector{Float64},dR::Float64,lowest_sig::Float64)
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
    return cube,model_cube
end

function create_model_cube(data::Dict{String, Any}, IX::Vector{Int64}, XX::Vector{Float64},IY::Vector{Int64},YY::Vector{Float64},Sd::Function,Sdpars::Vector{Float64},Rd::Vector{Float64},dR::Float64,disks::Disks;search_v=1,beam=nothing,lowest_sig=3.0)::Array{Float64, 3}
	cube=copy(data["data"])
	model_cube=fill(0.0,size(cube))
	Threads.@threads for n in 1:length(IX)
		ix=IX[n]
		x=XX[n]
		iy=IY[n]
		y=YY[n]
        cube,model_cube=manipulate_cube(ix,x,iy,y,cube,model_cube,data,disks,search_v,Sd,Sdpars,Rd,dR,lowest_sig)
	end
	if ~isnothing(beam)
		for iv in eachindex(data["V"])
			model_cube[:,:,iv]=imfilter(model_cube[:,:,iv],beam)
		end
	end
	return model_cube
end

# function create_model_cube(data::Dict{String, Any}, IX::Vector{Int64}, XX::Vector{Float64},IY::Vector{Int64},YY::Vector{Float64},Sd::Function,Sdpars::Vector{Float64},Rd::Vector{Float64},dR::Float64,disks::Disks;search_v=1,beam=nothing,lowest_sig=3.0)::Array{Float64, 3}
# 	cube=copy(data["data"])
# 	model_cube=fill(0.0,size(cube))
# 	@progress for n in 1:length(IX)
# 		ix=IX[n]
# 		x=XX[n]
# 		iy=IY[n]
# 		y=YY[n]
# 		for (rd,i_d,phi_d,V_d) in zip(disks.R,disks.I,disks.PA,disks.V)
# 			d,xy_ellipse=distf(rd, i_d, phi_d, [x,y])
# 			if d<=dR
# 				phi_ell=atan(y,x)#xy_ellipse[2],xy_ellipse[1])
# 				Vd_sky=V_d*sin(i_d)*sin(phi_ell-phi_d)/sqrt(1.0 +tan(i_d)^2.0*cos(phi_ell-phi_d)^2.0)
# 				loc=abs.(Vd_sky.-data["V"]) .<search_v
# 				if any(loc)
# 					lsic=cube[ix,iy,:]
# 					local_sig=lsic[loc]
# 					Vlocal=data["V"][loc]
# 					Fdata=maximum(local_sig)
# 					Vmax=Vlocal[argmax(local_sig)]
# 					if Fdata > lowest_sig*data["rms"]
# 						msig=Fdata .*exp.(-(data["V"] .-Vmax) .^2.0 ./(2.0*Sd(rd,Sdpars...)^2.0))
# 						model_cube[ix,iy,:]+=msig
# 						cube[ix,iy,:] -=msig
# 					end
# 				end
# 			end
# 		end
# 	end
# 	if ~isnothing(beam)
# 		for iv in eachindex(data["V"])
# 			model_cube[:,:,iv]=imfilter(model_cube[:,:,iv],beam)
# 		end
# 	end
# 	return model_cube
# end
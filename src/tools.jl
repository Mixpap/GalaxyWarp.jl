function Gaussian2DKernel(sx::Float64,sy::Float64,t::Float64)::Matrix{Float64}
	a=cos(t)^2.0/(2.0*sx^2.0)+sin(t)^2.0/(2.0*sy^2.0)
	b=sin(2.0*t)/(2.0*sx^2.0)-sin(2.0*t)/(2.0*sy^2.0)
	c=sin(t)^2.0/(2.0*sx^2.0)+cos(t)^2.0/(2.0*sy^2.0)
	A=1.0/(2.0*pi*sx*sy)
	L=4.0*max(sx,sy)
	xx=collect(-L:1:L)
	yy=xx
	return [A*exp(-a*x^2.0-b*x*y-c*y^2.0) for x in xx, y in yy]
end


function load_cube(cube_fits,name;z=0.0,rms=0.0,centerx="median",centery="median",beam="header",zaxis="freq",line_rest=230.538e9,vel_rest=0.0,x0=-5.0,x1=5.0,y0=-5.0,y1=5.0,v0=-500.0,v1=500.0)
    f = FITS(cube_fits)
    header = read_header(f[1])
    cosmo = cosmology()
    D = luminosity_dist(cosmo, z)

    data = read(f[1])
    if size(data)[end]==1
        data=data[:, :, :, 1]
    end
    rad_Mpc = angular_diameter_dist(cosmo, z) / u"rad"
    deg2kpc = uconvert(u"kpc" / u"°", rad_Mpc)

    Nx = header["NAXIS1"]
    dx = abs(header["CDELT1"])
    b_x =header["CRVAL1"]-header["CRPIX1"]*dx
    x=LinRange(b_x+dx,b_x+dx*Nx,Nx)u"°"
    if isnothing(centerx)
        x_center=header["RA"]u"°"
    elseif centerx=="median"
        x_center=median(x)
    else
        x_center=centerx
    end
    x=x.-x_center
    x_kpc=x .*deg2kpc
    xx = ustrip(x_kpc)

    Ny = header["NAXIS2"]
    dy = abs(header["CDELT2"])
    b_y =header["CRVAL2"]-header["CRPIX2"]*dy
    y=LinRange(b_y+dy,b_y+dy*Ny,Ny)u"°"
    if isnothing(centerx)
        y_center=header["DEC"]u"°"
    elseif centery=="median"
        y_center=median(y)
    else
        y_center=centery
    end
    y=y.-y_center
    y_kpc=y .*deg2kpc
    yy = ustrip(y_kpc)
    if beam == "header"
        bmaj=(header["BMAJ"]u"°")*deg2kpc
        bmin=(header["BMIN"]u"°")*deg2kpc
        bpa=header["BPA"]
        beamr=[ustrip(bmaj),ustrip(bmin),bpa]
    else
        beamr=beam #in kpc
    end
    dx=diff(xx)[1]
    b1=round(beamr[1]/dx /2.35)
	b2=round(beamr[2]/dx /2.35)
	b_pa_rad=deg2rad(beamr[3]+90.0) #correction for north
	psf= Gaussian2DKernel(b1,b2,b_pa_rad);

    if zaxis == "freq"
        dnu=header["CDELT3"]u"Hz"
        b_nu=header["CRVAL3"]u"Hz"-header["CRPIX3"]*dnu
        Nnu=header["NAXIS3"]
        nu=LinRange(b_nu+dnu,b_nu+dnu*Nnu,Nnu)
        line_rest = (line_rest)u"Hz" / (1.0 + z)
        v =((line_rest^2.0 .- nu .^ 2.0) ./ (line_rest^2.0 .+ nu .^ 2.0) .*c_0)#u"km" / u"s"
        v=v .|> u"km"/u"s"
        vv=[ustrip(vi) for vi ∈ v]
    elseif (zaxis == "vel")
        Nv = header["NAXIS3"]
        dv = abs(header["CDELT3"])
        b_v =header["CRVAL3"]-header["CRPIX3"]*dv
        v=LinRange(b_v+dv,b_v+dv*Nv,Nv)
        v=v.-vel_rest
        #if !(header["CUNIT3"]=="km/s")
        vv=v ./ 1000.0
        #else
        #    vv=v
        #end
    else 
        @error "Cannot understand the z coordinate"
    end
    #masking
    fv=findall(x -> (x .> v0) .& (x .< v1), vv)
    fx=findall(x -> (x .> x0) .& (x .< x1), xx)
    fy=findall(x -> (x .> y0) .& (x .< y1), yy)

    return Dict("name"=>name,"X"=>xx[fx],"Y"=>yy[fy],"V"=>vv[fv],"dx"=>diff(xx)[1],"dv"=>diff(vv)[1],"data"=>Float64.(data[fx,fy,fv]),"rms"=>rms,"beam"=>beamr,"BEAM"=>psf,"header"=>header)
end



F(θ,X,Y,var,;slit=0.1)= var[(Y .> -slit/(2.0*cos(θ-pi/2.0)) .+ X .*tan(θ-pi/2.0)) .& (Y .< slit/(2.0*cos(θ-pi/2.0)) .+ X .*tan(θ-pi/2.0))]

function pvd(ang,data;slit=0.1)
    slit_pix=Int(round(slit/data["dx"]))
	icx=Int(round(size(data["data"])[1]/2))
    A=zeros(size(data["data"]))
    for i in 1:size(data["data"])[3]
        A[:,:,i]=collect(imrotate(data["data"][:,:,i],deg2rad(ang),size(data["data"][:,:,1]),fill=0.0))[:,:,1]
    end
    return sum(A[icx-slit_pix:icx+slit_pix,:,:], dims=1)[1,:,:] ./(2.0*slit_pix*data["rms"])
end

function Clouds_Mass(I,data;n=2,v0=115.271,aco=4.2)
    beam_area=data["beam"][1]*data["beam"][2]*pi/(4.0*log(2.0)) #kpc^2
    pixel_area=data["dx"]^2.0
    #Ip=clouds.Fp .* clouds.Sp .* sqrt(2.0*pi) .*(pixel_area/beam_area) #jy km/s
    Mp=(3.25e7*aco*D^2.0/(n*v0)^2.0/(1.0+z)) .*I
    return Mp
end

function Outflow_Rate(M,V,R)
    Vkm_yr=V/3.168808781402895e-08
    R_km=R*3.0856775814671916e+16
    return Vkm_yr*M/R_km
end

function Outflow_Power(M,V,R)
    dotM=Outflow_Rate(M,V,R)
    return 0.5*V^2.0*dotM*6.300890659296177e+35
end
function Outflow_Momentum(M,V,R)
    dotM=Outflow_Rate(M,V,R)
    return V*dotM*6.300890659296178e+30
end
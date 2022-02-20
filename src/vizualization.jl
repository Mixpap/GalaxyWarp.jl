function plot_pvd(data,ang;cdata=nothing,clouds=nothing,disks=nothing,merged=false,slit=0.1,zoom_pvd=[-4.1,4.1,-370.0,370.0,1.0,50.0,150.0])
    θ=deg2rad(ang)
    ϕ=θ+pi/2
    figpvf=Figure(resolution=(1.9*600,650))
    axpvf=Axis(figpvf[1,1], xlabel=L"Projected radius [$\mathrm{kpc}$]",ylabel=L"Projected Velocity [$\mathrm{km\,s^{-1}}$]", labelsize=20)
    contourf!(axpvf, data["Y"], data["V"], pvd(ang,data;slit=slit), levels=collect(2:2:20), colormap=:roma,extendhigh=:red)

    #heatmap!(axpvf,data["Y"],data["V"],pvd(ang,data;slit=slit))
    if !isnothing(clouds)
        X=merged ? clouds.Xc : clouds.Xp
        Y=merged ? clouds.Yc : clouds.Yp
        V=merged ? clouds.Vc : clouds.Vp
        S=merged ? clouds.Sc : clouds.Sp

        Φ=atan.(Y,X)
        Rp=sqrt.(Y .^2.0 .+X .^2.0 )
        R_project=F(θ,X,Y,Rp;slit=slit).* (-sign.(cos.(F(θ,X,Y,Φ;slit=slit) .- (θ-pi/2.0))))
        Vslit=F(θ,X,Y,V;slit=slit)
        dVslit=F(θ,X,Y,S;slit=slit)
        chm=GLMakie.errorbars!(R_project, Vslit,dVslit, markersize=10)
    end

    if !isnothing(disks)
        for (n,di) in enumerate(keys(disks))
			r_model_m=Vector{Float64}()
			v_model_m=Vector{Float64}()
			r_model=Vector{Float64}()
			v_model=Vector{Float64}()
			
			for (rd,i_d,phi_d,v_d) in zip(disks[di].R, disks[di].I, disks[di].PA,disks[di].V)
				Rd_sky=rd/sqrt(1.0+tan(i_d)^2.0*cos(ϕ-phi_d)^2.0)
				Vd_sky=v_d*sin(i_d)*sin(ϕ-phi_d)/sqrt(1.0 +tan(i_d)^2.0*cos(ϕ-phi_d)^2.0)
				push!(r_model_m,-Rd_sky)
				push!(v_model_m,-Vd_sky)
				push!(r_model,Rd_sky)
				push!(v_model,Vd_sky)
			end
			lines!(axpvf,r_model_m,v_model_m,color=Cycled(n),linewidth=4)
			lines!(axpvf,r_model,v_model,color=Cycled(n),linewidth=4,label=di)
		end
		axislegend(axpvf)
    end
    axpvf.xticks = -37:zoom_pvd[5]:37
    axpvf.yticks =	-500:zoom_pvd[6]:500
    xlims!(axpvf,zoom_pvd[1],zoom_pvd[2])
    ylims!(axpvf,zoom_pvd[3],zoom_pvd[4])
    return figpvf
end

function plot_disks(disks;sample=nothing)
	fig=Figure()
	axpa=Axis(fig[1, 1])
	axi=Axis(fig[2, 1])
	
	for di in keys(disks)
		lines!(axpa,disks[di].R,rad2deg.(disks[di].PA),label=di)
		lines!(axi,disks[di].R,rad2deg.(disks[di].I),label=di)
	end
	axislegend(axpa)
	axislegend(axi)
	# if !isnothing(sample)
	# 	alpha_pvd=7.0/size(sample)[1]
	# 	for ppi in 1:size(sample)[1]
	# 		pp_i=sample[ppi,:]
	# 		update_parameters!(P,pp_i)
	# 		disks_tmp = make_disks(P.Rd,P)
	# 		lines!(axpa,disks_tmp.R,rad2deg.(disks_tmp.PA), transparency=true, color=RGBA(0.0,0.0,0.0,alpha_pvd))
	# 		lines!(axi,disks_tmp.R,rad2deg.(disks_tmp.I), transparency=true, color=RGBA(0.0,0.0,0.0,alpha_pvd))
	# 	end
	# end
	axpa.ylabel="Position Angle [deg]"
	axi.ylabel="Inclination [deg]"
	axi.xlabel="Radius [kpc]"
	axpa.xticks = 0.0:0.5:16.0
	axi.xticks = 0.0:0.5:16.0
	fig
end

function plot_disksv(disks;sample=nothing)
	fig=Figure()
	axpa=Axis(fig[1, 1])
	axi=Axis(fig[2, 1])
	
	for di in keys(disks)
		lines!(axpa,disks[di].R,disks[di].V,label=di)
	end
	axislegend(axpa)
	# if !isnothing(sample)
	# 	alpha_pvd=7.0/size(sample)[1]
	# 	for ppi in 1:size(sample)[1]
	# 		pp_i=sample[ppi,:]
	# 		update_parameters!(P,pp_i)
	# 		disks_tmp = make_disks(P.Rd,P)
	# 		lines!(axpa,disks_tmp.R,rad2deg.(disks_tmp.PA), transparency=true, color=RGBA(0.0,0.0,0.0,alpha_pvd))
	# 		lines!(axi,disks_tmp.R,rad2deg.(disks_tmp.I), transparency=true, color=RGBA(0.0,0.0,0.0,alpha_pvd))
	# 	end
	# end
	axpa.ylabel="Rotational Velocity [km/s]"
	axpa.xlabel="Radius [kpc]"
	axpa.xticks = 0.0:0.5:16.0
	fig
end




function plot_sky(geom::Geometry,disks::Disks;msize=10,zoom_image=[-2.0,2.0,-5.0,5.0,1.0,1.0])
    imratio=(zoom_image[2]-zoom_image[1])/(zoom_image[4]-zoom_image[3])
    figsky=Figure(resolution=(imratio*800,820))
	#ai = show_data ? 2 : 1
	axsky = Axis(figsky[2, 1],xgridvisible=true,ygridvisible=true)
	axsky.aspect = DataAspect()
    cmap=((:blue, 0.95), (:red, 0.95))
    sc=scatter!(axsky,geom.X,geom.Y,color=geom.dV,colorrange=(-80,80),markersize=msize,marker=:diamond,transparency=true,colormap=:bluesreds,label="CO(2-1) residuals")
    Colorbar(figsky[1,1], sc,vertical=false,label=L"dV [$\mathrm{km\,s^{-1}}$]")

    phis=0.0:0.01:2*pi
	for (rdi,i_di,pa_di) in zip(disks.R,disks.I,disks.PA)
        Rd_sky=rdi ./(1.0 .+tan.(i_di)^2.0 .* cos.(phis .-pa_di).^2.0).^0.5
        xd=Rd_sky .*cos.(phis)
        yd=Rd_sky .*sin.(phis)
        lines!(axsky,xd,yd,transparency = true, color=RGBA(0.0,0.0,0.0,0.1))
	end
    axsky.xticks = -4:zoom_image[5]:4
	axsky.yticks =	-5:zoom_image[6]:5
	xlims!(axsky,zoom_image[1],zoom_image[2])
	ylims!(axsky,zoom_image[3],zoom_image[4])
	axsky.xlabel="X [kpc]"
	axsky.ylabel="Y [kpc]"
    return figsky
end

function plot_cube_model(data,model;disks=nothing,zoom_image=[-2.0,2.0,-5.0,5.0,1.0,1.0])
    imratio=(zoom_image[2]-zoom_image[1])/(zoom_image[4]-zoom_image[3])

    figd=Figure(resolution=(imratio*800,820))
    axd = Axis(figd[2, 1],xgridvisible=true,ygridvisible=true,ylabel="Y [kpc]",xlabel="X [kpc]")
    axd.aspect = DataAspect()

    momm=sum(model["data"],dims=3)[:,:,1]
    data_cube=copy(data["data"])
    momd=sum(data_cube,dims=3)[:,:,1]

    maxmask=findall(x->x < 4*data["rms"],maximum(data["data"],dims=3)[:,:,1])
    momd[maxmask] .= NaN

    rres=(momm .- momd) ./(data["rms"]*size(data["data"])[3])
    sc=GLMakie.contourf!(axd,model["X"],model["Y"], rres,levels=[-2,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,1.25,1.5,2],colormap=:bluesreds)
    GLMakie.Colorbar(figd[1,1],sc,label="(Model-Data)/(rms * channels)",vertical=false,fontsize=25)
    
    if !isnothing(disks)
        phis_=collect(0:0.01:2*pi)
        for (rdi,i_di,pa_di) in zip(disks.R,disks.I,disks.PA)
            Rd_sky=rdi ./(1.0 .+tan.(i_di)^2.0 .* cos.(phis_ .-pa_di).^2.0).^0.5
            xd=Rd_sky .*cos.(phis_)
            yd=Rd_sky .*sin.(phis_)
            lines!(axd,xd,yd, label=false, transparency = true, color=RGBA(0.0,0.0,0.0,0.06))
        end
    end
    
    axd.xticks = -4:zoom_image[5]:4
	axd.yticks =	-5:zoom_image[6]:5
	xlims!(axd,zoom_image[1],zoom_image[2])
	ylims!(axd,zoom_image[3],zoom_image[4])
	axd.xlabel="X [kpc]"
	axd.ylabel="Y [kpc]"
    figd
end
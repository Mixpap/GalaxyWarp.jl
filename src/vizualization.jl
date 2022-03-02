
function moments(data,rx,ry;x0=-2.2,x1=2.2,y0=-4.5,y1=4.5,v0=-400.0,v1=400.0,dx=1.0,dv=20.0,vmax=80.0,denoise=4.0,show_beam=true)
	figmom=Figure(resolution = (rx,ry))
    fv=findall(x -> (x .> v0) .& (x .< v1), data["V"])
    fx=findall(x -> (x .> x0) .& (x .< x1), data["X"])
    fy=findall(x -> (x .> y0) .& (x .< y1), data["Y"])

    cube=data["data"][fx,fy,fv]
	#cube=filter(!isnan, cube)
    cube[cube .< (denoise * data["rms"])] .= 0.0
    X=data["X"][fx]
    Y=data["Y"][fy]
    V=data["V"][fv]

    axmom0= Axis(figmom[2,1])
    axmom0.xticks = collect(-10:dx:10)
    axmom0.yticks = collect(-10:dx:10)
    axmom0.aspect=DataAspect()
    mom0=sum(cube,dims=3)[:,:,1]
    mom0[mom0 .== 0] .= NaN
    hm=heatmap!(axmom0,X,Y,mom0)
    Colorbar(figmom[1, 1], hm, vertical = false,label=L"Flux $[\mathrm{Jy\,km\,s^{1}\,beam^{-1}}]$")
    axmom0.xlabel="X [kpc]"
    axmom0.ylabel="Y [kpc]"

    axmom1= Axis(figmom[2,2])
    axmom1.xticks = collect(-10:dx:10)
    axmom1.yticks = collect(-10:dx:10)
    axmom1.aspect=DataAspect()
    mom0=sum(cube,dims=3)[:,:,1]
    mom1=[sum(V .* cube[ix,iy,:]) for ix in 1:size(cube,1), iy in 1:size(cube,2)] ./mom0
    hm=heatmap!(axmom1,X,Y,mom1,colorrange=[v0,v1],colormap=:seismic)
    cb=Colorbar(figmom[1, 2], hm, vertical = false,label=L"Mean Velocity $[\mathrm{km\,s^{1}}]$")
    cb.ticks = (v0:100:v1)
    axmom1.xlabel="X [kpc]"
    axmom1.ylabel="Y [kpc]"

    axmom2= Axis(figmom[2,3])
    axmom2.xticks = collect(-10:dx:10)
    axmom2.yticks = collect(-10:dx:10)
    axmom2.aspect=DataAspect()
    mom0=sum(cube,dims=3)[:,:,1]
    mom1=[sum(V .* cube[ix,iy,:]) for ix in 1:size(cube,1), iy in 1:size(cube,2)] ./mom0
    mom2=sqrt.([sum((V .- mom1[ix,iy]) .^2.0 .* cube[ix,iy,:]) for ix in 1:size(cube,1), iy in 1:size(cube,2)] ./mom0)
    hm=heatmap!(axmom2,X,Y,mom2,colorrange=[0.0,vmax])#,colormap=:seismic)
    cb=Colorbar(figmom[1, 3], hm, vertical = false,label=L"Velocity Dispersion $[\mathrm{km\,s^{1}}]$")
    cb.ticks = (0.0:20.0:vmax)
    axmom2.xlabel="X [kpc]"
    axmom2.ylabel="Y [kpc]"
    #rowsize!(figmom.layout, 1, 0.95*axmom2.scene.px_area[].widths[2])
	#colsize!(figmom.layout, 1, 0.95*axmom2.scene.px_area[].widths[1])
	if show_beam
		phis_beam=0.0:0.01:2*pi
		major_beam=maximum(data["beam"][1:2])/2.0
		minor_beam=minimum(data["beam"][1:2])/2.0
		X_beam=0.6
		Y_beam=0.25
		i_beam=acos(minor_beam/major_beam)
		pa_beam=deg2rad(data["beam"][3])
		R_beam=major_beam ./(1.0 .+tan.(i_beam)^2.0 .* cos.(phis_beam .-pa_beam).^2.0).^0.5
		x_beam=R_beam .*cos.(phis_beam) .+0.8 *x1
		y_beam=R_beam .*sin.(phis_beam) .+ (y0+abs(0.1*y0))
		lines!(axmom0,x_beam,y_beam, label=false, transparency = true,linewith=4.0, color=:blue)
	end

    return figmom
end

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


function plot_sky(clouds::Clouds,disks::Disks;msize=10,zoom_image=[-2.0,2.0,-5.0,5.0,1.0,1.0])
    imratio=(zoom_image[2]-zoom_image[1])/(zoom_image[4]-zoom_image[3])
    figsky=Figure(resolution=(imratio*800,820))
	#ai = show_data ? 2 : 1
	axsky = Axis(figsky[2, 1],xgridvisible=true,ygridvisible=true)
	axsky.aspect = DataAspect()
    cmap=((:blue, 0.95), (:red, 0.95))
    sc=scatter!(axsky,clouds.Xp,clouds.Yp,color=clouds.dV,colorrange=(-80,80),markersize=msize,marker=:diamond,transparency=true,colormap=:bluesreds,label="CO(2-1) residuals")
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

function cloud_fitting_diagnostics(data,clouds::Clouds,savename;sigma=4.0,pvds=[0.0,90.0],slit=0.1)
    mom0_data = sum(data["data"] .> sigma*data["rms"], dims = 3)[:, :, 1];

    Xp=clouds.Xp
    Yp=clouds.Yp
    Vp=clouds.Vp
    Ip=clouds.Ip
    Sp=clouds.Sp
    Rp=sqrt.(Xp .^2 .+ Yp .^2)
    Φp=atan.(Yp,Xp)

    figd=Figure(resolution = (1500,700))
    axRc= Axis(figd[1,1],Aspect=:equal)
    axxy= Axis(figd[1,2],Aspect=:equal)
    axS= Axis(figd[1,3],Aspect=:equal)
    
    heatmap!(axxy,data["X"],data["Y"],mom0_data)
    lines!(axRc,Rp[sortperm(Rp)],log10.(cumsum(Ip[sortperm(Rp)])))
    scatter!(axxy,Xp,Yp,color=Vp,colorrange=[-300.0,300.0],colormap=:seismic,markersize=3,trasnparency=true)
    
    scatter!(axS,Rp,Sp,markersize=2,trasnparency=true)
    if length(clouds.Xc)<1
        save("$(savename)_fit.png",figd)
    end
    # for pvd in pvds
    #     save("$(savename)_pvd_$(pvd).png",plot_pvd(data,pvd;clouds=clouds,merged=false))
    # end

    if length(clouds.Xc)>1
        Xc=clouds.Xc
        Yc=clouds.Yc
        Vc=clouds.Vc
        Ic=clouds.Ic
        Sc=clouds.Sc
        Rc=sqrt.(Xc .^2 .+ Yc .^2)
        Φc=atan.(Yc,Xc)

        lines!(axRc,Rc[sortperm(Rc)],log10.(cumsum(Ic[sortperm(Rc)])))
        #scatter!(axxy,Xc,Yc,color=Vc,colorrange=[-300.0,300.0],colormap=:seismic,markersize=10,trasnparency=true)
        
        scatter!(axS,Rc,Sc,markersize=10,trasnparency=true)
        save("$(savename)_fit.png",figd)
        

        figd=Figure(resolution = (1500,700))
        axxy1= Axis(figd[1,1],Aspect=:equal)
        axxy= Axis(figd[1,2],Aspect=:equal)
        scatter!(axxy1,Xc,Yc,color=Vc,colorrange=[-300.0,300.0],colormap=:seismic,markersize=10,trasnparency=true)
        for (n,i) in enumerate(unique(clouds.I))
            color=Cycled(n)
            xx=Xp[clouds.belongs .== i]
            yy=Yp[clouds.belongs .== i]
            scatter!(axxy,xx,yy,color=color,markersize=8)
        end
        scatter!(axxy,Xc,Yc,color=:black,marker='X',markersize=12)
        save("$(savename)_merging.png",figd)

        for pvdi in pvds
            θ=deg2rad(pvdi)
            figpvf=Figure(resolution=(1.9*1000,1050))
            axpvf=Axis(figpvf[1,1], xlabel=L"Projected radius [$\mathrm{kpc}$]",ylabel=L"Projected Velocity [$\mathrm{km\,s^{-1}}$]", labelsize=20)
            try
                contourf!(axpvf, data["Y"], data["V"], pvd(pvdi,data;slit=slit), levels=collect(2:2:20), colormap=:roma,extendhigh=:red)
            catch er
                @warn "Cannot make contour plots, trying heatmap"
                heatmap!(axpvf, data["Y"], data["V"], pvd(pvdi,data;slit=slit),colorrange=[2,20],colormap=:roma,extendhigh=:red)
            end
            for (n,i) in enumerate(unique(clouds.I))
                color=Cycled(n)
                xx=Xp[clouds.belongs .== i]
                yy=Yp[clouds.belongs .== i]
                vv=Vp[clouds.belongs .== i]

                rc=sqrt.(xx .^2.0 .+ yy .^2.0)
                phic=atan.(yy,xx)
                rr_p=F(θ,xx,yy,rc;slit=slit).* (-sign.(cos.(F(θ,xx,yy,phic;slit=slit) .- (θ-pi/2.0))))

                vv_p=F(θ,xx,yy,vv;slit=slit)

                scatter!(axpvf,rr_p,vv_p,color=color,markersize=3,transparency=true)

                xx=[Xp[i]]
                yy=[Yp[i]]
                vv=[Vp[i]]
                rc=sqrt.(xx .^2.0 .+ yy .^2.0)
                phic=atan.(yy,xx)
                rr_p=F(θ,xx,yy,rc;slit=slit).* (-sign.(cos.(F(θ,xx,yy,phic;slit=slit) .- (θ-pi/2.0))))

                vv_p=F(θ,xx,yy,vv;slit=slit)

                scatter!(axpvf,rr_p,vv_p,color=:black,marker=:rect,markersize=13,transparency=true)
            end
            xx=Xp
            yy=Yp
            vv=Vp
            rc=sqrt.(xx .^2.0 .+ yy .^2.0)
            phic=atan.(yy,xx)
            rr_p=F(θ,xx,yy,rc;slit=slit).* (-sign.(cos.(F(θ,xx,yy,phic;slit=slit) .- (θ-pi/2.0))))

            vv_p=F(θ,xx,yy,vv;slit=slit)

            scatter!(axpvf,rr_p,vv_p,color=:black,marker=:rect,markersize=2,transparency=true)

            #scatter!(axpvf,F(θ,Xc,Yc,Rc;slit=slit).* (-sign.(cos.(F(θ,Xc,Yc,Φc;slit=slit) .- (θ-pi/2.0)))),F(θ,Xc,Yc,Vc;slit=slit),color=:black,marker='X',markersize=12)

            save("$(savename)_pvd_$(pvdi)_merged.png",figpvf)
        end

        # figb=Figure(resolution = (2000,1000))
        # axxyb= Axis(figb[1,1],Aspect=:equal)
        # axyvb = Axis(figb[1,2],Aspect=:equal)
        # scatter!(axxyb,Xp,Yp,color=clouds.belongs,makersize=7)
        # scatter!(axyvb,Yp,Vp,color=clouds.belongs,makersize=7)
        # for (n,i) in enumerate(clouds.I)
        #     scatter!(axxyb,[Xp[i]],[Yp[i]],color=:red,marker="$(n%9)",markersize=15)
        #     scatter!(axyvb,[Yp[i]],[Vp[i]],color=:red,marker="$(n%9)",markersize=15)
        # end
        # save("$(savename)_fit_merging.png",figb)
    end
end

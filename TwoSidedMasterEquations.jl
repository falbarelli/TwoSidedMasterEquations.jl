module TwoSidedMasterEquations

using LinearAlgebra
using QuantumOptics
using QuantumOpticsBase

export master_twosided, master_twosided_dynamic, calc_abstr, calc_tracenorm, calc_tracenorm_and_abstr

function dmaster_twosided!(drho, Hl, Hr, Jl, Jldagger, Jr, Jrdagger, rho, drho_cache)
    
    QuantumOpticsBase.mul!(drho,Hl,rho,-eltype(rho)(im),zero(eltype(rho))) # drho = -im*Hl*rho
    QuantumOpticsBase.mul!(drho,rho,Hr,eltype(rho)(im),one(eltype(rho))) # drho = im*rho*Hr + drho
    # So far: drho = -im*Hl*rho + im*rho*Hr
    
    for i=1:length(Jl)
        QuantumOpticsBase.mul!(drho_cache,Jl[i],rho) # drho_cache = Jl*rho
        QuantumOpticsBase.mul!(drho,drho_cache,Jrdagger[i],true,true) # drho = drho_cache*Jrdagger + drho
        # So far: drho = -im*Hl*rho + im*rho*Hr + Jl*rho*Jrdagger
        
        QuantumOpticsBase.mul!(drho,Jldagger[i],drho_cache,eltype(rho)(-0.5),one(eltype(rho))) # drho = -0.5*Jldagger*drho_cache + drho
        # So far: drho = -im*Hl*rho + im*rho*Hr + Jl*rho*Jrdagger - 0.5*Jldagger*Jl*rho
        
        QuantumOpticsBase.mul!(drho_cache,rho,Jrdagger[i],true,false) # drho_cache = rho*Jrdagger 
        QuantumOpticsBase.mul!(drho,drho_cache,Jr[i],eltype(rho)(-0.5),one(eltype(rho))) # drho = -0.5*drho_cache*Jr + drho
        # Finally: drho = -im*Hl*rho + im*rho*Hr + Jl*rho*Jrdagger - 0.5*Jldagger*Jl*rho - 0.5*rho*Jrdagger*Jr
    end
    
    return drho
end

function master_twosided(tspan, rho0::Operator, Hl::AbstractOperator,Hr::AbstractOperator, Jl, Jr; 
					Jldagger=dagger.(Jl),
					Jrdagger=dagger.(Jl),
					fout=nothing,
					kwargs...)
		
	tmp = copy(rho0)
    
	dmaster_h_ts(t, rho, drho) = dmaster_twosided!(drho, Hl, Hr, Jl, Jldagger, Jr, Jrdagger, rho, tmp)
        
	return QuantumOptics.timeevolution.integrate_master(tspan, dmaster_h_ts, rho0, fout; kwargs...)
	
end

function master_twosided_dynamic(tspan, rho0::Operator, f::Function; 
					fout=nothing,
					kwargs...)
	# `f`: Function `f(t, rho) -> (Hl , Hr, Jl, Jr)`
        
    tmp = copy(rho0)
    
	function dmaster_h_ts(t, rho, drho) 
        Hl, Hr, Jl, Jr = f(t, rho)
        Jldagger=dagger.(Jl)
        Jrdagger=dagger.(Jr)
        return dmaster_twosided!(drho, Hl, Hr, Jl, Jldagger, Jr, Jrdagger, rho, tmp)
    end
        
	return QuantumOptics.timeevolution.integrate_master(tspan, dmaster_h_ts, rho0, fout; kwargs...)
	
end

@eval $:master_twosided(tspan,psi0::Ket,args...;kwargs...) = $:master_twosided(tspan,dm(psi0),args...;kwargs...)
@eval $:master_twosided_dynamic(tspan,psi0::Ket,args...;kwargs...) = $:master_twosided_dynamic(tspan,dm(psi0),args...;kwargs...)

# callout functions, to be used as fout=... when calling master_twosided() or master_twosided_dynamic()
function calc_tracenorm_and_abstr(t, ρ)
    return tracenorm(ρ) , abs(tr(ρ))
end

function calc_tracenorm(t, ρ)
    return tracenorm(ρ)
end

function calc_abstr(t, ρ)
    return abs(tr(ρ))
end


end

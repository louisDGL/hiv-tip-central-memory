"""Parameter table. Single source of truth for every number in the project.

Each field carries its source in a comment. Fields marked ASSUMPTION are not
measured and must be reported as assumptions in the paper, not as data.

Nothing downstream should hard-code a numeric value that lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # --- inherited from Weinberger 2003, Table 1 -------------------------
    lam: float = 3.1e4        # cells mL^-1 d^-1   (31 cells uL^-1 d^-1)
    d: float = 0.02           # d^-1   loss rate, uninfected target
    beta_H: float = 1.875e-7  # mL virion^-1 d^-1  infection rate per virion
    beta_T: float = 1.875e-7  # mL virion^-1 d^-1  = beta_H (TIP phenotypically identical)
    delta: float = 0.7        # d^-1   loss rate, I_H
    N: float = 200.0          # virions cell^-1    burst size
    c: float = 30.0           # d^-1   virion clearance
    D: float = 0.8            # -      inhibition of HIV output from I_D (1 - D_W, D_W ~ 0.2)
    Pi: float = 12.0          # -      packaging advantage (P^2 in Weinberger notation)

    # --- the design layer -------------------------------------------------
    A: float = 12.0           # -      TIP output per superinfection.
    #   Pitchai 2024 Fig 2F, three-colour coculture. CEILING, NOT A MEASUREMENT:
    #   the assay starts from producer cells already transduced and then
    #   infected, so superinfection is arranged rather than sampled. Sweep it.
    A_sd: float = 0.5

    rho: float = 0.015        # d^-1   self-renewal, central memory
    #   Macallan 2004 J Exp Med 200:255, Table II (1.5%/day, CD45R0+ CCR7+ CD4+)

    phi: float = 0.0          # d^-1   per-cell differentiation inflow into T_CM
    #   UNMEASURED IN HUMANS. phi = 0 is the "purely self-renewing" case and
    #   forces d_M = rho at steady state. phi > 0 means the compartment is fed
    #   from stem-memory, and then d_M = rho + phi.
    #
    #   The human T_SCM -> T_CM flux has never been quantified. Costa Del Amo
    #   2018 PLoS Biol 16:e2005523 is the only human T_SCM kinetic model and
    #   does not model T_CM at all; Gattinoni 2011 Nat Med 17:1290 shows the
    #   pathway qualitatively only. The one quantitative analogue is MOUSE:
    #   Gossel 2017 eLife 6:e23013 puts naive -> T_CM influx at ~1.5%/day,
    #   i.e. phi ~ rho. See PHI_MOUSE below and analysis/gate_dm_vs_rho.py

    eps: float = 0.0          # -      fitness cost of carrying the TIP.
    #   UNMEASURED. The object of Aim 3. No published experiment bounds it.

    # --- clinical anchors --------------------------------------------------
    V0: float = 1e5           # copies mL^-1   HIV set point, pre-therapy
    V_target: float = 1e3     # copies mL^-1   ASSUMPTION: one log10 below the
    #   SHIV set point reached under TIP; above the 400 copies mL^-1 failure
    #   threshold cited in Weinberger 2003
    mu_ref: float = 5.2e-4    # d^-1   net decay, latent reservoir
    #   Siliciano 2003 Nat Med 9:727, t_1/2 = 44.2 months

    # --- structural switch -------------------------------------------------
    superinfect_IH: bool = True
    #   True  -> I_D is fed by both routes, as written in M1/M2 section 2:
    #            I_H + V_T  and  I_T + V_H.
    #   False -> only I_T + V_H, which is what the PUBLISHED code does
    #            (HIV-TIP/expandedHiv.m has no -k*Vt*I term in dI).
    #   Must be False to verify against the paper's own figures. M2 section 3
    #   already derives the threshold on the single route only, calling the
    #   bound conservative, so the threshold itself is unaffected.

    # ---------------------------------------------------------------------
    # Derived quantities. None of these are free parameters.
    # ---------------------------------------------------------------------

    @property
    def delta_D(self) -> float:
        """Loss rate of dually infected cells. (1 - D) * delta = 0.14 /d.
        Cancels from the persistence threshold."""
        return (1.0 - self.D) * self.delta

    @property
    def d_M(self) -> float:
        """Loss rate of the central-memory compartment.

        At steady state the untransduced compartment satisfies
        rho * M + inflow = d_M * M, so d_M = rho + phi. With phi = 0 this is
        the labelled assumption d_M = rho.
        """
        return self.rho + self.phi

    @property
    def T_star(self) -> float:
        """CD4+ target cells at the HIV-only equilibrium, c / (N * beta_H)."""
        return self.c / (self.N * self.beta_H)

    @property
    def A_derived(self) -> float:
        """A = Pi * N * beta_T * T_star / c reduces to Pi * beta_T / beta_H."""
        return self.Pi * self.beta_T / self.beta_H

    @property
    def mu(self) -> float:
        """Net loss rate of the TIP-carrying central-memory compartment, /d.

        mu = d_M - rho * (1 - eps) = phi + rho * eps.

        This is the one quantity that has to come out right. The design enters
        the published floor through mu and nothing else.
        """
        return self.d_M - self.rho * (1.0 - self.eps)

    def V_floor(self, A: float | None = None) -> float:
        """Viral load below which the TIP goes extinct, copies mL^-1.

        V_H_floor = mu / (beta_H * (A - 1)).
        """
        A = self.A if A is None else A
        return self.mu / (self.beta_H * (A - 1.0))

    def mu_max(self, A: float | None = None, V_target: float | None = None) -> float:
        """Largest mu that still holds the load at V_target: (A-1) * beta_H * V_target."""
        A = self.A if A is None else A
        V = self.V_target if V_target is None else V_target
        return (A - 1.0) * self.beta_H * V

    def eps_crit(self, A: float | None = None, V_target: float | None = None) -> float:
        """Largest admissible fitness cost, given phi.

        From mu = phi + rho*eps <= mu_max:   eps <= (mu_max - phi) / rho.
        Negative means infeasible at any eps: the inflow alone breaks the bound.
        """
        return (self.mu_max(A, V_target) - self.phi) / self.rho

    def p_superinfection(self, V_H: float | None = None) -> float:
        """Probability a TIP-only cell is superinfected before it is lost.

        p = beta_H*V_H / (d_M + beta_H*V_H - rho*(1-eps)) for the memory
        compartment; the published short-lived version uses d in place of the
        net rate. This is the cumulative probability over the cell's life, NOT
        the prevalence of multiply-infected cells at one instant. Aim 1 is the
        conversion between the two.
        """
        V_H = self.V0 if V_H is None else V_H
        force = self.beta_H * V_H
        return force / (force + self.mu)

    def R0_TIP(self, V_H: float | None = None, A: float | None = None) -> float:
        """R0^TIP = p * A."""
        A = self.A if A is None else A
        return self.p_superinfection(V_H) * A

    def with_(self, **kw) -> "Params":
        """Return a copy with fields overridden. p2 = p.with_(eps=0.05)."""
        return replace(self, **kw)


#: Mouse naive -> T_CM influx, Gossel 2017 eLife 6:e23013, Table 1.
#: 9.8%/week (6.5-16.9) = 1.4%/day. The only quantitative analogue for phi.
#: NOT a human number. Use it to bracket, never as the value.
PHI_MOUSE: float = 0.014

#: Posterior median of A from the base paper's own MCMC fit, A = P^2.
#: See analysis/a_from_posterior.py. Table 1's A = 12 comes from Fig 2F
#: instead, and the two disagree by a factor of ~4.8.
A_POSTERIOR_MEDIAN: float = 2.50
A_POSTERIOR_95: tuple = (2.065, 3.426)

#: The published baseline: purely self-renewing compartment, no fitness cost,
#: A from Fig 2F. This is M2 Table 1 as written.
BASELINE = Params()

#: The same design read against the base paper's own fitted A.
FITTED_A = Params(A=A_POSTERIOR_MEDIAN, Pi=A_POSTERIOR_MEDIAN)

#: Weinberger 2003 as published: short-lived TIP-only cell, no maintenance,
#: single route into I_D. Recovering this is the verification target (Aim 2).
PUBLISHED = Params(rho=0.0, phi=0.02, superinfect_IH=False)

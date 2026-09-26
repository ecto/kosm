// Gauge-field operations shared by the kernels. Each kernel module declares
// `p` (its Params) and `links` (read or read_write) itself.

fn load(mu: u32, site: u32) -> M3 {
    var m: M3;
    let b = (mu * p.n + site) * 9u;
    for (var k = 0u; k < 9u; k++) {
        m.e[k] = links[b + k];
    }
    return m;
}

// Sum of the six staples around U_mu(site), oriented so the local action is
// −(β/3) Re Tr(U A).
fn staple(site: u32, mu: u32) -> M3 {
    return staple_from(site, mu, 0u);
}

// Staples from directions nu ≥ `first` only: first = 1 keeps the time
// direction out, for spatial-only smearing.
fn staple_from(site: u32, mu: u32, first: u32) -> M3 {
    let n_mu = fwd(p, site, mu);
    var a = m_zero();
    for (var nu = first; nu < 4u; nu++) {
        if (nu == mu) {
            continue;
        }
        let n_nu = fwd(p, site, nu);
        a = m_add(a, m_mul_dag(m_mul_dag(load(nu, n_mu), load(mu, n_nu)), load(nu, site)));
        let n_mnu = bwd(p, site, nu);
        let n_mu_mnu = bwd(p, n_mu, nu);
        a = m_add(a, m_mul(m_dag_mul(load(nu, n_mu_mnu), m_dag(load(mu, n_mnu))), load(nu, n_mnu)));
    }
    return a;
}

fn plaquette(site: u32, mu: u32, nu: u32) -> M3 {
    let a = m_mul(load(mu, site), load(nu, fwd(p, site, mu)));
    let b = m_mul(load(nu, site), load(mu, fwd(p, site, nu)));
    return m_mul_dag(a, b);
}

// Sum of the four plaquette leaves in the (mu, nu) plane around `site`.
fn clover(site: u32, mu: u32, nu: u32) -> M3 {
    let n_mu = fwd(p, site, mu);
    let n_nu = fwd(p, site, nu);
    let n_mmu = bwd(p, site, mu);
    let n_mnu = bwd(p, site, nu);
    let n_mmu_nu = bwd(p, n_nu, mu);
    let n_mmu_mnu = bwd(p, n_mmu, nu);
    let n_mu_mnu = fwd(p, n_mnu, mu);
    let l1 = m_mul_dag(m_mul(load(mu, site), load(nu, n_mu)), m_mul(load(nu, site), load(mu, n_nu)));
    let l2 = m_mul(m_mul_dag(m_mul_dag(load(nu, site), load(mu, n_mmu_nu)), load(nu, n_mmu)), load(mu, n_mmu));
    let l3 = m_dag_mul(m_mul(load(nu, n_mmu_mnu), load(mu, n_mmu)), m_mul(load(mu, n_mmu_mnu), load(nu, n_mnu)));
    let l4 = m_mul(m_dag_mul(load(nu, n_mnu), load(mu, n_mnu)), m_mul_dag(load(nu, n_mu_mnu), load(mu, site)));
    return m_add(m_add(l1, l2), m_add(l3, l4));
}

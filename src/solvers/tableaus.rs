// Runge-Kutta tableau registry — data-only definitions.
//
// Each solver's Butcher tableau lives here as a `pub const Tableau`.
// Factories in `factories.rs` consume these to configure a `Solver` instance.
//
// Marker convention: an empty `bt[i] = &[]` denotes an explicit ESDIRK stage
// (maps to the historical `Option::None` row).  Empty `tr`/`a_final` mean
// "not present".

/// Solver family — used by capability checks (e.g. which problem forms the
/// tableau can integrate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableauKind {
    /// Explicit Runge-Kutta.
    ExplicitRK,
    /// Diagonally implicit Runge-Kutta (all stages implicit).
    DIRK,
    /// Explicitly-first Diagonally implicit RK (first stage explicit).
    ESDIRK,
}

/// A complete Runge-Kutta tableau plus associated metadata.
#[derive(Debug, Clone, Copy)]
pub struct Tableau {
    pub name: &'static str,
    pub kind: TableauKind,
    /// Classical order.
    pub n: usize,
    /// Embedded order (0 if non-adaptive).
    pub m: usize,
    /// Number of stages.
    pub s: usize,
    /// Stage time offsets (length s).
    pub eval_stages: &'static [f64],
    /// Butcher A/b matrix.  bt[i] = coefficients for stage i.
    /// bt[i] = &[] denotes an explicit stage (ESDIRK first stage).
    pub bt: &'static [&'static [f64]],
    /// Truncation residual coefficients for adaptive step control.
    /// Empty slice = not adaptive.
    pub tr: &'static [f64],
    /// Non-stiffly-accurate final-row coefficients.
    /// Empty slice = last bt row is also the output row.
    pub a_final: &'static [f64],
}

impl Tableau {
    #[inline] pub const fn is_adaptive(&self) -> bool { !self.tr.is_empty() }
    #[inline] pub const fn is_explicit(&self) -> bool { matches!(self.kind, TableauKind::ExplicitRK) }
    #[inline] pub const fn is_implicit(&self) -> bool { !self.is_explicit() }
}

// ======================================================================================
// Explicit fixed-step
// ======================================================================================

pub const SSPRK22: Tableau = Tableau {
    name: "SSPRK22",
    kind: TableauKind::ExplicitRK,
    n: 2, m: 0, s: 2,
    eval_stages: &[0.0, 1.0],
    bt: &[&[1.0], &[0.5, 0.5]],
    tr: &[], a_final: &[],
};

pub const RK4: Tableau = Tableau {
    name: "RK4",
    kind: TableauKind::ExplicitRK,
    n: 4, m: 0, s: 4,
    eval_stages: &[0.0, 0.5, 0.5, 1.0],
    bt: &[
        &[0.5],
        &[0.0, 0.5],
        &[0.0, 0.0, 1.0],
        &[1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0],
    ],
    tr: &[], a_final: &[],
};

pub const SSPRK33: Tableau = Tableau {
    name: "SSPRK33",
    kind: TableauKind::ExplicitRK,
    n: 3, m: 0, s: 3,
    eval_stages: &[0.0, 1.0, 0.5],
    bt: &[
        &[1.0],
        &[0.25, 0.25],
        &[1.0/6.0, 1.0/6.0, 2.0/3.0],
    ],
    tr: &[], a_final: &[],
};

pub const SSPRK34: Tableau = Tableau {
    name: "SSPRK34",
    kind: TableauKind::ExplicitRK,
    n: 3, m: 0, s: 4,
    eval_stages: &[0.0, 0.5, 1.0, 0.5],
    bt: &[
        &[0.5],
        &[0.5, 0.5],
        &[1.0/6.0, 1.0/6.0, 1.0/6.0],
        &[1.0/6.0, 1.0/6.0, 1.0/6.0, 0.5],
    ],
    tr: &[], a_final: &[],
};

// ======================================================================================
// Explicit adaptive
// ======================================================================================

pub const RKF21: Tableau = Tableau {
    name: "RKF21",
    kind: TableauKind::ExplicitRK,
    n: 2, m: 1, s: 3,
    eval_stages: &[0.0, 0.5, 1.0],
    bt: &[
        &[0.5],
        &[1.0/256.0, 255.0/256.0],
        &[1.0/512.0, 255.0/256.0, 1.0/512.0],
    ],
    tr: &[1.0/512.0, 0.0, -1.0/512.0],
    a_final: &[],
};

pub const RKBS32: Tableau = Tableau {
    name: "RKBS32",
    kind: TableauKind::ExplicitRK,
    n: 3, m: 2, s: 4,
    eval_stages: &[0.0, 0.5, 0.75, 1.0],
    bt: &[
        &[0.5],
        &[0.0, 0.75],
        &[2.0/9.0, 1.0/3.0, 4.0/9.0],
        &[2.0/9.0, 1.0/3.0, 4.0/9.0],
    ],
    tr: &[-5.0/72.0, 1.0/12.0, 1.0/9.0, -1.0/8.0],
    a_final: &[],
};

pub const RKF45: Tableau = Tableau {
    name: "RKF45",
    kind: TableauKind::ExplicitRK,
    n: 5, m: 4, s: 6,
    eval_stages: &[0.0, 0.25, 3.0/8.0, 12.0/13.0, 1.0, 0.5],
    bt: &[
        &[0.25],
        &[3.0/32.0, 9.0/32.0],
        &[1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0],
        &[439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0],
        &[-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0],
        // Propagate the 5th-order weights b5 (local extrapolation, like RKCK54/
        // RKDP54).  Fehlberg's 4th-order row b4 = [25/216, 0, 1408/2565,
        // 2197/4104, -1/5, 0] was previously advanced here while `n: 5` claimed
        // 5th order, capping fixed-step convergence at 4 (issue #23).  `tr`
        // stays b5 - b4 and now estimates the lower-order solution's error.
        &[16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0],
    ],
    tr: &[1.0/360.0, 0.0, -128.0/4275.0, -2197.0/75240.0, 1.0/50.0, 2.0/55.0],
    a_final: &[],
};

pub const RKCK54: Tableau = Tableau {
    name: "RKCK54",
    kind: TableauKind::ExplicitRK,
    n: 5, m: 4, s: 6,
    eval_stages: &[0.0, 0.2, 0.3, 0.6, 1.0, 7.0/8.0],
    bt: &[
        &[0.2],
        &[3.0/40.0, 9.0/40.0],
        &[0.3, -0.9, 1.2],
        &[-11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0],
        &[1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0],
        &[37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0],
    ],
    tr: &[-277.0/64512.0, 0.0, 6925.0/370944.0, -6925.0/202752.0, -277.0/14336.0, 277.0/7084.0],
    a_final: &[],
};

pub const RKDP54: Tableau = Tableau {
    name: "RKDP54",
    kind: TableauKind::ExplicitRK,
    n: 5, m: 4, s: 7,
    eval_stages: &[0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0],
    bt: &[
        &[0.2],
        &[3.0/40.0, 9.0/40.0],
        &[44.0/45.0, -56.0/15.0, 32.0/9.0],
        &[19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
        &[9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
        &[35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0],
        &[35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0],
    ],
    tr: &[71.0/57600.0, 0.0, -71.0/16695.0, 71.0/1920.0, -17253.0/339200.0, 22.0/525.0, -1.0/40.0],
    a_final: &[],
};

pub const RKV65: Tableau = Tableau {
    name: "RKV65",
    kind: TableauKind::ExplicitRK,
    n: 6, m: 5, s: 9,
    eval_stages: &[0.0, 9.0/50.0, 1.0/6.0, 0.25, 53.0/100.0, 0.6, 0.8, 1.0, 1.0],
    bt: &[
        &[9.0/50.0],
        &[29.0/324.0, 25.0/324.0],
        &[1.0/16.0, 0.0, 3.0/16.0],
        &[79129.0/250000.0, 0.0, -261237.0/250000.0, 19663.0/15625.0],
        &[1336883.0/4909125.0, 0.0, -25476.0/30875.0, 194159.0/185250.0, 8225.0/78546.0],
        &[-2459386.0/14727375.0, 0.0, 19504.0/30875.0, 2377474.0/13615875.0, -6157250.0/5773131.0, 902.0/735.0],
        &[2699.0/7410.0, 0.0, -252.0/1235.0, -1393253.0/3993990.0, 236875.0/72618.0, -135.0/49.0, 15.0/22.0],
        &[11.0/144.0, 0.0, 0.0, 256.0/693.0, 0.0, 125.0/504.0, 125.0/528.0, 5.0/72.0],
        &[11.0/144.0, 0.0, 0.0, 256.0/693.0, 0.0, 125.0/504.0, 125.0/528.0, 5.0/72.0],
    ],
    // tr = b - b_hat (b is bt[8] = [11/144, 0, 0, 256/693, 0, 125/504, 125/528, 5/72, 0],
    // b_hat = [28/477, 0, 0, 212/441, -312500/366177, 2125/1764, 0, -2105/35532, 2995/17766])
    tr: &[
        11.0/144.0 - 28.0/477.0,
        0.0,
        0.0,
        256.0/693.0 - 212.0/441.0,
        0.0 - (-312500.0/366177.0),
        125.0/504.0 - 2125.0/1764.0,
        125.0/528.0,
        5.0/72.0 - (-2105.0/35532.0),
        0.0 - 2995.0/17766.0,
    ],
    a_final: &[],
};

pub const RKF78: Tableau = Tableau {
    name: "RKF78",
    kind: TableauKind::ExplicitRK,
    n: 7, m: 8, s: 13,
    eval_stages: &[0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 0.5, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0],
    bt: &[
        &[2.0/27.0],
        &[1.0/36.0, 1.0/12.0],
        &[1.0/24.0, 0.0, 1.0/8.0],
        &[5.0/12.0, 0.0, -25.0/16.0, 25.0/16.0],
        &[1.0/20.0, 0.0, 0.0, 0.25, 0.2],
        &[-25.0/108.0, 0.0, 0.0, 125.0/108.0, -65.0/27.0, 125.0/54.0],
        &[31.0/300.0, 0.0, 0.0, 0.0, 61.0/225.0, -2.0/9.0, 13.0/900.0],
        &[2.0, 0.0, 0.0, -53.0/6.0, 704.0/45.0, -107.0/9.0, 67.0/90.0, 3.0],
        &[-91.0/108.0, 0.0, 0.0, 23.0/108.0, -976.0/135.0, 311.0/54.0, -19.0/60.0, 17.0/6.0, -1.0/12.0],
        &[2383.0/4100.0, 0.0, 0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0],
        &[3.0/205.0, 0.0, 0.0, 0.0, 0.0, -6.0/41.0, -3.0/205.0, -3.0/41.0, 3.0/41.0, 6.0/41.0],
        &[-1777.0/4100.0, 0.0, 0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0],
        &[41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0],
    ],
    tr: &[41.0/840.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.0/840.0, -41.0/840.0, -41.0/840.0],
    a_final: &[],
};

pub const RKDP87: Tableau = Tableau {
    name: "RKDP87",
    kind: TableauKind::ExplicitRK,
    n: 8, m: 7, s: 13,
    eval_stages: &[0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0, 93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0, 1201146811.0/1299019798.0, 1.0, 1.0],
    bt: &[
        &[1.0/18.0],
        &[1.0/48.0, 1.0/16.0],
        &[1.0/32.0, 0.0, 3.0/32.0],
        &[5.0/16.0, 0.0, -75.0/64.0, 75.0/64.0],
        &[3.0/80.0, 0.0, 0.0, 3.0/16.0, 3.0/20.0],
        &[29443841.0/614563906.0, 0.0, 0.0, 77736538.0/692538347.0, -28693883.0/1125000000.0, 23124283.0/1800000000.0],
        &[16016141.0/946692911.0, 0.0, 0.0, 61564180.0/158732637.0, 22789713.0/633445777.0, 545815736.0/2771057229.0, -180193667.0/1043307555.0],
        &[39632708.0/573591083.0, 0.0, 0.0, -433636366.0/683701615.0, -421739975.0/2616292301.0, 100302831.0/723423059.0, 790204164.0/839813087.0, 800635310.0/3783071287.0],
        &[246121993.0/1340847787.0, 0.0, 0.0, -37695042795.0/15268766246.0, -309121744.0/1061227803.0, -12992083.0/490766935.0, 6005943493.0/2108947869.0, 393006217.0/1396673457.0, 123872331.0/1001029789.0],
        &[-1028468189.0/846180014.0, 0.0, 0.0, 8478235783.0/508512852.0, 1311729495.0/1432422823.0, -10304129995.0/1701304382.0, -48777925059.0/3047939560.0, 15336726248.0/1032824649.0, -45442868181.0/3398467696.0, 3065993473.0/597172653.0],
        &[185892177.0/718116043.0, 0.0, 0.0, -3185094517.0/667107341.0, -477755414.0/1098053517.0, -703635378.0/230739211.0, 5731566787.0/1027545527.0, 5232866602.0/850066563.0, -4093664535.0/808688257.0, 3962137247.0/1805957418.0, 65686358.0/487910083.0],
        &[403863854.0/491063109.0, 0.0, 0.0, -5068492393.0/434740067.0, -411421997.0/543043805.0, 652783627.0/914296604.0, 11173962825.0/925320556.0, -13158990841.0/6184727034.0, 3936647629.0/1978049680.0, -160528059.0/685178525.0, 248638103.0/1413531060.0, 0.0],
        &[14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0, 181606767.0/758867731.0, 561292985.0/797845732.0, -1041891430.0/1371343529.0, 760417239.0/1151165299.0, 118820643.0/751138087.0, -528747749.0/2220607170.0, 0.25],
    ],
    // tr = bt[12] - b_hat, where
    // b_hat = [13451932/455176623, 0, 0, 0, 0, -808719846/976000145, 1757004468/5645159321,
    //          656045339/265891186, -3867574721/1518517206, 465885868/322736535,
    //          53011238/667516719, 2/45, 0]
    tr: &[
        14005451.0/335480064.0 - 13451932.0/455176623.0,
        0.0, 0.0, 0.0, 0.0,
        -59238493.0/1068277825.0 - (-808719846.0/976000145.0),
        181606767.0/758867731.0 - 1757004468.0/5645159321.0,
        561292985.0/797845732.0 - 656045339.0/265891186.0,
        -1041891430.0/1371343529.0 - (-3867574721.0/1518517206.0),
        760417239.0/1151165299.0 - 465885868.0/322736535.0,
        118820643.0/751138087.0 - 53011238.0/667516719.0,
        -528747749.0/2220607170.0 - 2.0/45.0,
        0.25,
    ],
    a_final: &[],
};

// ======================================================================================
// Implicit fixed-step (DIRK / ESDIRK)
// ======================================================================================

pub const DIRK2: Tableau = Tableau {
    name: "DIRK2",
    kind: TableauKind::DIRK,
    n: 2, m: 0, s: 2,
    eval_stages: &[0.25, 0.75],
    bt: &[&[0.25], &[0.5, 0.25]],
    tr: &[],
    a_final: &[0.5, 0.5],
};

pub const DIRK3: Tableau = Tableau {
    name: "DIRK3",
    kind: TableauKind::DIRK,
    n: 3, m: 0, s: 4,
    eval_stages: &[0.5, 2.0/3.0, 0.5, 1.0],
    bt: &[
        &[0.5],
        &[1.0/6.0, 0.5],
        &[-0.5, 0.5, 0.5],
        &[1.5, -1.5, 0.5, 0.5],
    ],
    tr: &[], a_final: &[],
};

pub const ESDIRK4: Tableau = Tableau {
    name: "ESDIRK4",
    kind: TableauKind::ESDIRK,
    n: 4, m: 0, s: 6,
    eval_stages: &[0.0, 0.5, 1.0/6.0, 37.0/40.0, 0.5, 1.0],
    bt: &[
        &[],
        &[0.25, 0.25],
        &[-1.0/36.0, -1.0/18.0, 0.25],
        &[-21283.0/32000.0, -5143.0/64000.0, 90909.0/64000.0, 0.25],
        &[46010759.0/749250000.0, -737693.0/40500000.0, 10931269.0/45500000.0, -1140071.0/34090875.0, 0.25],
        &[89.0/444.0, 89.0/804756.0, -27.0/364.0, -20000.0/171717.0, 843750.0/1140071.0, 0.25],
    ],
    tr: &[], a_final: &[],
};

// ======================================================================================
// Implicit adaptive (ESDIRK)
// ======================================================================================

pub const ESDIRK32: Tableau = Tableau {
    name: "ESDIRK32",
    kind: TableauKind::ESDIRK,
    n: 3, m: 2, s: 4,
    eval_stages: &[0.0, 1.0, 1.5, 1.0],
    bt: &[
        &[],
        &[0.5, 0.5],
        &[5.0/8.0, 3.0/8.0, 0.5],
        &[7.0/18.0, 1.0/3.0, -2.0/9.0, 0.5],
    ],
    tr: &[-1.0/9.0, -1.0/6.0, -2.0/9.0, 0.5],
    a_final: &[],
};

const ESDIRK43_G: f64 = 0.25;  // diagonal entry

pub const ESDIRK43: Tableau = Tableau {
    name: "ESDIRK43",
    kind: TableauKind::ESDIRK,
    n: 4, m: 3, s: 6,
    eval_stages: &[
        0.0,
        0.5,
        // (2 - sqrt(2)) / 4 — baked in as a float literal so the slice stays const.
        0.146_446_609_406_726_2,
        2012122486997.0 / 3467029789466.0,
        1.0,
        1.0,
    ],
    bt: &[
        &[],
        &[ESDIRK43_G, ESDIRK43_G],
        &[-1356991263433.0/26208533697614.0, -1356991263433.0/26208533697614.0, ESDIRK43_G],
        &[-1778551891173.0/14697912885533.0, -1778551891173.0/14697912885533.0, 7325038566068.0/12797657924939.0, ESDIRK43_G],
        &[-24076725932807.0/39344244018142.0, -24076725932807.0/39344244018142.0, 9344023789330.0/6876721947151.0, 11302510524611.0/18374767399840.0, ESDIRK43_G],
        &[657241292721.0/9909463049845.0, 657241292721.0/9909463049845.0, 1290772910128.0/5804808736437.0, 1103522341516.0/2197678446715.0, -3.0/28.0, ESDIRK43_G],
    ],
    // tr = a1 - a2 (main minus embedded)
    tr: &[
        657241292721.0/9909463049845.0 - (-71925161075.0/3900939759889.0),
        657241292721.0/9909463049845.0 - (-71925161075.0/3900939759889.0),
        1290772910128.0/5804808736437.0 - 2973346383745.0/8160025745289.0,
        1103522341516.0/2197678446715.0 - 3972464885073.0/7694851252693.0,
        -3.0/28.0 - (-263368882881.0/4213126269514.0),
        ESDIRK43_G - 3295468053953.0/15064441987965.0,
    ],
    a_final: &[],
};

// Kennedy/Carpenter 2019 "ESDIRK5(4)8L[2]SAb" (Appl. Numer. Math. 146:221-244).
// Upgrade from the 2003 7-stage ESDIRK5(4)7L[2]SA2: both main AND embedded
// estimators are now L-stable, fixing the over-rejection pathology of the
// 7-stage variant on very stiff problems (VdP mu=1000, Oregonator, ...).
// Coefficients cross-checked against SUNDIALS ARKODE
// `ARKODE_ARK548L2SAb_DIRK_8_4_5` (src/arkode/arkode_butcher_dirk.def).
const ESDIRK54_G: f64 = 2.0/9.0;

pub const ESDIRK54: Tableau = Tableau {
    name: "ESDIRK54",
    kind: TableauKind::ESDIRK,
    n: 5, m: 4, s: 8,
    eval_stages: &[
        0.0,
        4.0/9.0,
        6456083330201.0/8509243623797.0,
        1632083962415.0/14158861528103.0,
        6365430648612.0/17842476412687.0,
        18.0/25.0,
        191.0/200.0,
        1.0,
    ],
    bt: &[
        &[],
        &[ESDIRK54_G, ESDIRK54_G],
        &[2366667076620.0/8822750406821.0, 2366667076620.0/8822750406821.0, ESDIRK54_G],
        &[-257962897183.0/4451812247028.0, -257962897183.0/4451812247028.0, 128530224461.0/14379561246022.0, ESDIRK54_G],
        &[-486229321650.0/11227943450093.0, -486229321650.0/11227943450093.0, -225633144460.0/6633558740617.0, 1741320951451.0/6824444397158.0, ESDIRK54_G],
        &[621307788657.0/4714163060173.0, 621307788657.0/4714163060173.0, -125196015625.0/3866852212004.0, 940440206406.0/7593089888465.0, 961109811699.0/6734810228204.0, ESDIRK54_G],
        &[2036305566805.0/6583108094622.0, 2036305566805.0/6583108094622.0, -3039402635899.0/4450598839912.0, -1829510709469.0/31102090912115.0, -286320471013.0/6931253422520.0, 8651533662697.0/9642993110008.0, ESDIRK54_G],
        &[0.0, 0.0, 3517720773327.0/20256071687669.0, 4569610470461.0/17934693873752.0, 2819471173109.0/11655438449929.0, 3296210113763.0/10722700128969.0, -1142099968913.0/5710983926999.0, ESDIRK54_G],
    ],
    // tr = b − b_hat   (b = last A row; b_hat from KC19 Table 17 / SUNDIALS `d[]`)
    tr: &[
        0.0,
        0.0,
        3517720773327.0/20256071687669.0 - 520639020421.0/8300446712847.0,
        4569610470461.0/17934693873752.0 - 4550235134915.0/17827758688493.0,
        2819471173109.0/11655438449929.0 - 1482366381361.0/6201654941325.0,
        3296210113763.0/10722700128969.0 - 5551607622171.0/13911031047899.0,
        -1142099968913.0/5710983926999.0 - (-5266607656330.0/36788968843917.0),
        ESDIRK54_G - 1074053359553.0/5740751784926.0,
    ],
    a_final: &[],
};

// ======================================================================================
// Registry — iterate over every tableau.  Used by factory_from_name() and by
// capability reporting (e.g. Python `list_solvers()`).
// ======================================================================================

pub const ALL: &[&Tableau] = &[
    &SSPRK22, &RK4, &SSPRK33, &SSPRK34,
    &RKF21, &RKBS32, &RKF45, &RKCK54, &RKDP54, &RKV65, &RKF78, &RKDP87,
    &DIRK2, &DIRK3, &ESDIRK4,
    &ESDIRK32, &ESDIRK43, &ESDIRK54,
];

/// Find a tableau by its `name` (case-sensitive).
pub fn by_name(name: &str) -> Option<&'static Tableau> {
    ALL.iter().copied().find(|t| t.name == name)
}

/// All tableau-backed solver names (for diagnostics).
pub fn all_names() -> impl Iterator<Item = &'static str> {
    ALL.iter().map(|t| t.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tableau_dimensions_consistent() {
        for t in ALL {
            assert_eq!(t.eval_stages.len(), t.s, "{}: eval_stages length", t.name);
            assert_eq!(t.bt.len(), t.s, "{}: bt rows", t.name);
            for (i, row) in t.bt.iter().enumerate() {
                // Empty row marks ESDIRK explicit first stage.
                if row.is_empty() { continue; }
                // Regular Butcher row has i+1 entries.  FSAL-style tableaus
                // (RKBS32, RKDP54, RKV65, RKF78) omit the trailing zero on
                // the final stage, so `<= i+1` is the right bound.
                assert!(row.len() <= i + 1,
                    "{}: bt[{}] has {} entries, expected ≤ {}",
                    t.name, i, row.len(), i + 1);
            }
            if !t.tr.is_empty() {
                assert_eq!(t.tr.len(), t.s, "{}: tr length", t.name);
            }
        }
    }

    #[test]
    fn esdirk_first_stage_is_explicit() {
        for t in ALL {
            if matches!(t.kind, TableauKind::ESDIRK) {
                assert!(t.bt[0].is_empty(), "{}: ESDIRK first stage must be explicit", t.name);
            }
        }
    }

    #[test]
    fn by_name_finds_all() {
        for t in ALL {
            assert!(by_name(t.name).is_some(), "{} not found by name", t.name);
        }
        assert!(by_name("NotARealSolver").is_none());
    }

    #[test]
    fn esdirk43_g_literal_matches_formula() {
        // (2 - sqrt(2)) / 4 — sanity check on the baked-in eval_stages[2].
        let expected = (2.0 - std::f64::consts::SQRT_2) / 4.0;
        assert!((ESDIRK43.eval_stages[2] - expected).abs() < 1e-15);
    }

    /// Fixed-step convergence-order fit for RKF45.  With the propagated row set
    /// to the 5th-order weights b5 (issue #23), the global-error slope on a
    /// smooth scalar ODE must be ~5; the pre-fix 4th-order b4 gives ~4.
    #[test]
    fn rkf45_fixed_step_convergence_order_is_five() {
        // dx/dt = x, x(0) = 1  ->  exact x(1) = e.
        let mut f = |x: &[f64], _t: f64, out: &mut Vec<f64>| { out.clear(); out.push(x[0]); };
        let exact = std::f64::consts::E;

        let factory = crate::solvers::factories::rkf45_factory(1e-12, 0.0);
        // Drive the stepper directly for exactly `1/dt` steps so the run lands
        // precisely on t = 1 — this isolates the tableau's fixed-step order from
        // the standalone `integrate()` loop-guard behaviour.
        let dts = [0.1, 0.05, 0.025, 0.0125];
        let mut errs = Vec::new();
        for &dt in &dts {
            let mut s = factory(&[1.0]);
            let n_steps = (1.0_f64 / dt).round() as usize;
            let (mut g_buf, mut jac_buf, mut f_buf) = (Vec::new(), Vec::new(), Vec::new());
            let mut t = 0.0;
            for _ in 0..n_steps {
                s.take_step(&mut f, None, None, t, dt, 0, None, &mut g_buf, &mut jac_buf, &mut f_buf);
                t += dt;
            }
            errs.push((s.x[0] - exact).abs());
        }

        // Least-squares slope of log(err) vs log(dt) == empirical order.
        let n = dts.len() as f64;
        let xs: Vec<f64> = dts.iter().map(|d| d.ln()).collect();
        let ys: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
        let (sx, sy) = (xs.iter().sum::<f64>(), ys.iter().sum::<f64>());
        let sxx = xs.iter().map(|v| v * v).sum::<f64>();
        let sxy = xs.iter().zip(&ys).map(|(a, b)| a * b).sum::<f64>();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);

        eprintln!("RKF45 fixed-step order fit slope = {slope:.4}; errs = {errs:?}");
        assert!(
            slope > 4.6 && slope < 5.4,
            "RKF45 empirical convergence order {slope:.3} not ~5 (errors: {errs:?})"
        );
    }
}

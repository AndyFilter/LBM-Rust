use crate::{MyApp};
use std::fmt::Display;
use std::time::Instant;

static SIZE_X: usize = 100;
static SIZE_Y: usize = 200;
const DIRECTION_COUNT: usize = 9;
pub(crate) const MIN_DENSITY_VAL: f32 = 1.0;

#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum CellType {
    CellTypeNone = 0,
    CellTypeWall = 1,
    CellTypeCount = 2,
}

impl TryFrom<u8> for CellType {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == CellType::CellTypeNone as u8 => Ok(CellType::CellTypeNone),
            x if x == CellType::CellTypeWall as u8 => Ok(CellType::CellTypeWall),
            x if x == CellType::CellTypeCount as u8 => Ok(CellType::CellTypeCount),
            _ => Err(()),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum WallType {
    WallBounceBack,
    WallSymmetric,
    WallDirichlet,
    WallOpen,
}

impl TryFrom<u8> for WallType {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == WallType::WallBounceBack as u8 => Ok(WallType::WallBounceBack),
            x if x == WallType::WallSymmetric as u8 => Ok(WallType::WallSymmetric),
            x if x == WallType::WallDirichlet as u8 => Ok(WallType::WallDirichlet),
            x if x == WallType::WallOpen as u8 => Ok(WallType::WallOpen),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CellInfo {
    pub(crate) cell_type: CellType,
    pub(crate) wall_type: WallType,
    pub(crate) in_fn: [f32; DIRECTION_COUNT],
    eq_fn: [f32; DIRECTION_COUNT],
    pub(crate) out_fn: [f32; DIRECTION_COUNT],
    pub(crate) density: f32,
    pub(crate) velocity: [f32; 2],
}

impl Display for CellInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?};{};{}", self.out_fn, self.cell_type as u8, self.wall_type as u8)
        //write!(f, "{:?};{};{};{}", self.velocity, self.density, self.cell_type as u8, self.wall_type as u8) // Zamiana formatu na {vx,vy}{rho}
    }
}

impl CellInfo {
    fn new_boundary(density: f32, cell_type: CellType, wall_type: WallType) -> Self {
        Self {
            in_fn: [0f32; DIRECTION_COUNT],
            eq_fn: [0f32; DIRECTION_COUNT],
            out_fn: [0f32; DIRECTION_COUNT],
            density,
            cell_type,
            wall_type,
            velocity: [0f32; 2],
        }
    }

    fn new(density: f32, cell_type: CellType) -> Self {
        Self {
            in_fn: [0f32; DIRECTION_COUNT],
            eq_fn: [0f32; DIRECTION_COUNT],
            out_fn: [0f32; DIRECTION_COUNT],
            density,
            cell_type,
            wall_type: WallType::WallBounceBack,
            velocity: [0f32; 2],
        }
    }
}

impl Default for CellInfo {
    fn default() -> Self {
        Self::new(0f32, CellType::CellTypeNone)
    }
}

// None, Forrest, Terrain, Water, City, Fire, Burned
pub(crate) static PLOT_COLORS_U8: [[u8; 4]; 8usize] = [
    //[32, 32, 32, 255],
    [0, 0, 0, 255],
    [39, 100, 39, 255],
    //[39, 39, 39, 255],
    [220, 32, 32, 255],
    [1, 111, 185, 255],
    [160, 160, 160, 255],
    [90, 90, 100, 255],
    [254, 127, 45, 255],
    [39, 39, 39, 255],
];

impl Default for MyApp {
    fn default() -> Self {
        let mut sim = vec![vec![CellInfo::default(); SIZE_Y]; SIZE_X];

        //let neighbors = [(1, 0), (0, -1i32), (-1i32, 0), (0, 1)];
        for x in 0..sim.len() {
            for y in 0..sim[0].len() {
                if x <= 1 || y <= 1 || x >= sim.len() - 2 || y >= sim[0].len() - 2 {
                    //sim[x][y].density = 0f32;
                    sim[x][y].cell_type = CellType::CellTypeWall;
                    //if x < 2 || x > SIZE_N - 3 {
                    sim[x][y].wall_type = WallType::WallDirichlet;
                    //}

                    if y <= 1 {
                        sim[x][y].wall_type = WallType::WallSymmetric;
                    } else if y >= SIZE_Y - 3 {
                        sim[x][y].wall_type = WallType::WallBounceBack;
                    }
                    //
                    // if x < 2 {
                    //     sim[x][y].wall_type = WallType::WallDirichlet;
                    // }
                } else if x < SIZE_X / 4 {
                    // if random::<u8>() <= 255 {
                    //     sim[x][y].in_fn[0] = 1.0; // = 4f32;
                    //     sim[x][y].density = 1.0;
                    // }
                    // if random::<u8>() <= 100 {
                    //     //sim[x][y].cell_type = CellTypeGas;
                    //     sim[x][y].in_fn[(random::<u8>() % DIRECTION_COUNT as u8) as usize] = (random::<u8>() % 20u8) as f32;
                    //     sim[x][y].density = sim[x][y].in_fn.iter().sum();
                    // }
                    //sim[x][y].in_fn[0] = 1.0;
                    sim[x][y].density = 1.0;
                } else if (x >= SIZE_X / 2 - 2 && x <= SIZE_X / 2 + 1) && y > 2*SIZE_Y / 3 {
                    sim[x][y].cell_type = CellType::CellTypeWall;
                    sim[x][y].wall_type = WallType::WallBounceBack;
                    // if y > N / 2 - (N / 6) && y < N / 2 + (N / 6) {
                    //     sim[x][y].density = MIN_DENSITY_VAL;
                    // } else {
                    //     sim[x][y].cell_type = CellTypeWall;
                    // }

                    //sim[x][y].density = MIN_DENSITY_VAL;
                    //sim[x][y].density = 0.9f32;
                }
                // else if x > 2 && y > 2 && ((x - 2*N/3).pow(2) + (y - N/2).pow(2)) < 75 {
                //     sim[x][y].cell_type = CellTypeWall;
                // }
                else {
                    //sim[x][y].in_fn[0] = 1.0; // = 4f32;
                    sim[x][y].density = MIN_DENSITY_VAL;
                }
            }
        }

        let neighbors = [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        ];
        let weights = [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ];
        for x in 0..sim.len() {
            for y in 0..sim[0].len() {
                let mut cell = &mut sim[x][y];
                if cell.cell_type == CellType::CellTypeWall {
                    continue;
                }
                for (i, &e) in neighbors.iter().enumerate() {

                    cell.eq_fn[i] = weights[i] * cell.density;
                    cell.in_fn[i] = cell.eq_fn[i];
                }
            }
        }

        Self {
            sim,
            sim_step: 0,
            last_sim_step: Instant::now(),
            img_texture: None,
            magnification: (800 / SIZE_Y) as f32,
            velocity_view: false,
            is_playing: false,
        }
    }
}



pub fn step_sim(state: &mut Vec<Vec<CellInfo>>, sim_step: &mut usize) {
    //let now = Instant::now();
    //let mut new_state: Vec<Vec<CellInfo>> = vec![vec![CellInfo::default(); N]; N];

    let (n_x, n_y) = (state.len(), state[0].len());

    let mut gas_particles = 0f32;
    let mut full_cells = 0;

    let dt = 1f32;
    let tau = 1f32;

    //let neighbors = [(1, 0), (0, -1i32), (-1i32, 0), (0, 1)];
    // let neighbors = [
    //     (-1,1),(0,1),(1,1),(-1,0),(0,0),(1,0),(-1,-1),(0,-1),(1,-1),
    // ];
    // let neighbors = [
    //     (0,0), (1,0), (-1,0), (0,1), (0, -1), (1,1), (-1,1), (-1,-1), (1,-1)
    // ];
    // +1 / -1 index reflection sorted
    let neighbors = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    ];
    let weights = [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];
    //let neighbors = [(1, 0), (-1i32, 0), (0, 1), (0, -1i32)];

    // Collision
    for x in 1..n_x - 1 {
        for y in 1..n_y - 1 {
            let mut cell = &mut state[x][y];
            // state.par_iter_mut().enumerate().for_each(|(x, row)| {
            //     row.par_iter_mut().enumerate().filter(|(_, a)| a.cell_type != CellTypeWall).for_each(|(y, cell)| {
            // state.par_chunks_mut(20).for_each(|chunk| {
            //     for row in chunk.iter_mut() {
            //         for cell in row.iter_mut() {
            if cell.cell_type == CellType::CellTypeWall {
                continue;
            } else {
                // gas_particles += cell.density;
                // full_cells += 1;
            }
            //let u = (cell.velocity[0], cell.velocity[1]);
            cell.density = cell.in_fn.iter().sum();
            let mut u = neighbors
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |acc, (i, c)| {
                    (
                        acc.0 + cell.in_fn[i] * c.0 as f32,
                        acc.1 + cell.in_fn[i] * c.1 as f32,
                    )
                });
            u = (u.0 / cell.density, u.1 / cell.density);
            cell.velocity[0] = u.0;
            cell.velocity[1] = u.1;
            let u_mag = 1.5 * (u.0.powi(2) + u.1.powi(2));
            for (i, &e) in neighbors.iter().enumerate() {
                let (n_x, n_y) = (e.0 as f32, e.1 as f32);
                let eu = (n_x * u.0 + n_y * u.1);
                cell.eq_fn[i] =
                    weights[i] * cell.density * (1.0 + 3.0 * eu + 4.5 * eu.powi(2) - u_mag);
                cell.out_fn[i] = cell.in_fn[i] + (dt / tau) * (cell.eq_fn[i] - cell.in_fn[i]);
            }
        }
    }
    //});

    // Streaming
    for x in 1..n_x - 1 {
        for y in 1..n_y - 1 {
            if state[x][y].cell_type == CellType::CellTypeWall {
                continue;
            }
            for (i, &e) in neighbors.iter().enumerate() {
                let (new_x, new_y): (_, _) = ((x as i32 - e.0) as usize, (y as i32 - e.1) as usize);

                if state[new_x][new_y].cell_type == CellType::CellTypeWall {
                    // ----------------------- Boundary Conditions -----------------------
                    match state[new_x][new_y].wall_type {
                        WallType::WallBounceBack => {
                            // if e.0 == 1 || (e.1 == 1 && e.0 == 0) {
                            if i % 2 == 1 {
                                state[x][y].in_fn[i] = state[x][y].out_fn[(i + 1)];
                            } else {
                                state[x][y].in_fn[i] = state[x][y].out_fn[(i - 1)];
                            }
                        }
                        WallType::WallSymmetric => {
                            // (0, 0),  0   -
                            // (1, 0),  1   E
                            // (-1, 0), 2   W
                            // (0, 1),  3   S
                            // (0, -1), 4   N
                            // (1, 1),  5   SE
                            // (-1, -1),6   NW
                            // (1, -1), 7   NE
                            // (-1, 1), 8   SW
                            //let symmetric_mapping = [0,2,1,4,3,8,7,6,5];
                            let symmetric_mapping = [0, 2, 1, 4, 3, 7, 8, 5, 6];
                            state[x][y].in_fn[i] = state[x][y].out_fn[symmetric_mapping[i]];
                        }
                        WallType::WallDirichlet => {
                            //let mut ux = 0.0f32;
                            //if y < 20
                            // All walls (ex. 1)
                            //let ux = if new_x > 2 && new_x < n_x - 3 { if new_y < 2 {0.02} else {0.0} } else { (n_y-y) as f32 / n_y as f32 * 0.02 };
                            let mut ux = 0.0f32;
                            let mut top_sum = 0.0f32;
                            let mut out_rho = 0.0f32;
                            let mut uy = 0.0; //6.0 / out_rho * (cell.out_fn[1] - cell.out_fn[2] + cell.out_fn[6]);
                            let (C, N, E, S, W, NE, NW, SE, SW) = (
                                0usize, 4usize, 1usize, 3usize, 2usize, 7usize, 6usize, 5usize,
                                8usize,
                            );
                            if new_x < 2 || new_x > n_x - 3 {
                                // Vertical walls
                                //println!("test");
                                if new_x < 2 {
                                    // left wall (West)
                                    // state[x][y].in_fn[C] = state[x][y].out_fn[C];
                                    // state[x][y].in_fn[W] = state[x][y].out_fn[W];
                                    let cell = &state[x][y];
                                    ux = (n_y - y) as f32 / n_y as f32 * 0.02;
                                    top_sum = (cell.in_fn[C]
                                        + cell.in_fn[S]
                                        + cell.in_fn[N]
                                        + 2.0 * (cell.in_fn[W] + cell.in_fn[SW] + cell.in_fn[NW]));
                                    out_rho = top_sum / (1.0 - ux);

                                    uy = 0.0; //6.0 * (state[x][y].in_fn[N] - state[x][y].in_fn[S] + state[x][y].in_fn[NW] - state[x][y].in_fn[SW]) / out_rho / (5.0 + 3.0 * ux);
                                    state[x][y].in_fn[E] =
                                        state[x][y].in_fn[W] + 2.0 / 3.0 * out_rho * ux;
                                    state[x][y].in_fn[NE] =
                                        state[x][y].in_fn[SW] + (ux - uy) / 6.0 * out_rho;
                                    state[x][y].in_fn[SE] =
                                        state[x][y].in_fn[NW] + (ux + uy) / 6.0 * out_rho;
                                } else {
                                    // right wall (East)
                                    // state[x][y].in_fn[C] = state[x][y].out_fn[C];
                                    // state[x][y].in_fn[E] = state[x][y].out_fn[E];

                                    let cell = &state[x][y];
                                    top_sum = (cell.in_fn[C]
                                        + cell.in_fn[S]
                                        + cell.in_fn[N]
                                        + 2.0 * (cell.in_fn[E] + cell.in_fn[SE] + cell.in_fn[NE]));

                                    // Boundary Ex. 1
                                    //ux = (n_y - y) as f32 / n_y as f32 * 0.02;
                                    //out_rho = top_sum / (1.0 + ux);

                                    // Boundary Ex. 2
                                    out_rho = 1.0;
                                    ux = top_sum / out_rho - 1.0;

                                    uy = 0.0; //6.0 * (state[x][y].in_fn[N] - state[x][y].in_fn[S] +
                                    //state[x][y].in_fn[NE] - state[x][y].in_fn[SE]) / out_rho / (5.0 - 3.0 * ux);
                                    state[x][y].in_fn[W] =
                                        state[x][y].in_fn[E] - 2.0 / 3.0 * out_rho * ux;
                                    state[x][y].in_fn[SW] =
                                        state[x][y].in_fn[NE] - (ux - uy) / 6.0 * out_rho;
                                    state[x][y].in_fn[NW] =
                                        state[x][y].in_fn[SE] - (ux + uy) / 6.0 * out_rho;
                                }
                            } else {
                                // Horizontal walls
                                if new_y < 2 {
                                    // upper wall
                                    // state[x][y].in_fn[C] = state[x][y].out_fn[C];
                                    // state[x][y].in_fn[N] = state[x][y].out_fn[N];
                                    ux = -0.02;
                                    uy = 0.0;
                                    let cell = &state[x][y];
                                    top_sum = (cell.in_fn[C]
                                        + cell.in_fn[E]
                                        + cell.in_fn[W]
                                        + 2.0 * cell.in_fn[N]
                                        + 2.0 * cell.in_fn[NW]
                                        + 2.0 * cell.in_fn[NE]);
                                    out_rho = top_sum / (1.0 + uy);

                                    state[x][y].in_fn[N] =
                                        state[x][y].in_fn[S] + 2.0 / 3.0 * out_rho * uy;
                                    state[x][y].in_fn[NW] =
                                        state[x][y].in_fn[SE] + (ux - uy) / 6.0 * out_rho;
                                    state[x][y].in_fn[NE] =
                                        state[x][y].in_fn[NW] - (ux + uy) / 6.0 * out_rho;
                                } else {
                                    // lower wall
                                    ux = 0.0;
                                    uy = 0.0;
                                    // state[x][y].in_fn[C] = state[x][y].out_fn[C];
                                    // state[x][y].in_fn[S] = state[x][y].out_fn[S];
                                    let cell = &state[x][y];
                                    top_sum = (cell.in_fn[C]
                                        + cell.in_fn[E]
                                        + cell.in_fn[W]
                                        + 2.0 * cell.in_fn[S]
                                        + 2.0 * cell.in_fn[SW]
                                        + 2.0 * cell.in_fn[SE]);
                                    out_rho = top_sum / (1.0 + uy);

                                    uy = 0.0; //6.0 * (state[x][y].in_fn[N] - state[x][y].in_fn[S] + state[x][y].in_fn[NE] - state[x][y].in_fn[SE]) / out_rho / (5.0 - 3.0 * ux);
                                    state[x][y].in_fn[S] =
                                        state[x][y].in_fn[N] - 2.0 / 3.0 * out_rho * uy;
                                    state[x][y].in_fn[SW] =
                                        state[x][y].in_fn[NE] - (ux - uy) / 6.0 * out_rho;
                                    state[x][y].in_fn[SE] =
                                        state[x][y].in_fn[NW] + (ux + uy) / 6.0 * out_rho;
                                }
                            }

                            // Left wall (ex. 2)
                            //let ux = if x <= 2 {(n_y-y) as f32 / n_y as f32 * 0.02} else {0.0};
                            // let cell = &state[x][y];
                            // let top_sum = (cell.out_fn[0] + cell.out_fn[1] + cell.out_fn[2] +
                            //     2.0 * cell.out_fn[4] + 2.0 * cell.out_fn[6] + 2.0*cell.out_fn[7]);
                            //let u_n = top_sum / 0.95 + 1.0;
                            //let out_rho = top_sum / (1.0 - ux);
                            //let out_rho = if x >= n_x-3 {1.0} else {top_sum / (1.0 - ux)}; // Right wall (ex. 2)

                            // state[x][y].in_fn[1] = state[x][y].out_fn[2] + 2.0/3.0 * out_rho * ux;
                            // state[x][y].in_fn[5] = state[x][y].out_fn[6] + 1.0/6.0 * out_rho * ux;
                            // state[x][y].in_fn[7] = state[x][y].out_fn[8] + 1.0/6.0 * out_rho * ux;
                        }
                        WallType::WallOpen => {}
                    }

                    continue;
                }

                state[x][y].in_fn[i] = state[new_x][new_y].out_fn[i];

                // if x == 2 || x == n_x - 3 || y == 2 {
                //     state[x][y].velocity[1] = 0.0;
                // }
            }
        }
    }

    println!("elements: {gas_particles}, gas_cells: {full_cells}");

    *sim_step += 1;
    //*state.clone()
}
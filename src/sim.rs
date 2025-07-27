use crate::{MyApp};
use std::fmt::Display;
use std::ops::{Range, RangeBounds, RangeInclusive};
use std::time::Instant;
use crate::boundary::{BoundaryGroup, BoundaryType, Rect};

static SIZE_X: usize = 100;
static SIZE_Y: usize = 200;
const DIRECTION_COUNT: usize = 9;
pub(crate) const MIN_DENSITY_VAL: f32 = 1f32;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellInfo {
    pub cell_type: CellType,
    //pub boundary_type: BoundaryType,
    pub in_fn: [f32; DIRECTION_COUNT],
    pub eq_fn: [f32; DIRECTION_COUNT],
    pub out_fn: [f32; DIRECTION_COUNT],
    pub density: f32,
    pub velocity: [f32; 2],
}

impl Display for CellInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?};{}", self.out_fn, self.cell_type as u8)
        //write!(f, "{:?};{};{};{}", self.velocity, self.density, self.cell_type as u8, self.wall_type as u8) // Different formatting - {vx,vy}{rho}
    }
}

impl CellInfo {
    // fn new_boundary(density: f32, cell_type: CellType, boundary_type: BoundaryType) -> Self {
    //     Self {
    //         in_fn: [0f32; DIRECTION_COUNT],
    //         eq_fn: [0f32; DIRECTION_COUNT],
    //         out_fn: [0f32; DIRECTION_COUNT],
    //         density,
    //         cell_type,
    //         //boundary_type,
    //         velocity: [0f32; 2],
    //     }
    // }

    fn new(density: f32, cell_type: CellType) -> Self {
        Self {
            in_fn: [0f32; DIRECTION_COUNT],
            eq_fn: [0f32; DIRECTION_COUNT],
            out_fn: [0f32; DIRECTION_COUNT],
            density,
            cell_type,
            //boundary_type: BoundaryType::WallBounceBack,
            velocity: [0f32; 2],
        }
    }
}

impl Default for CellInfo {
    fn default() -> Self {
        Self::new(0f32, CellType::CellTypeNone)
    }
}

// Space, Boundary
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

// +1 / -1 index reflection sorted
const NEIGHBORS: [(i32, i32); 9] = [
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
const WEIGHTS: [f32; 9] = [
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

impl Default for MyApp {
    fn default() -> Self {
        let mut sim = vec![vec![CellInfo::default(); SIZE_Y]; SIZE_X];

        // Add bounceback boundary around
        let mut b_conds: Vec<BoundaryGroup> = Vec::new();
        // Horizontal flow
        // b_conds.push(BoundaryGroup::new(Rect::new_all(0i32, 0i32, SIZE_X as i32, 1i32), BoundaryType::WallDirichlet, 0f32)); // Top wall
        // b_conds.push(BoundaryGroup::new(Rect::new_all(0, SIZE_Y as i32 - 1, SIZE_X as i32, 1), BoundaryType::WallDirichlet, 0f32)); // Bottom wall
        // b_conds.push(BoundaryGroup::new(Rect::new_all(0, 0, 1, SIZE_Y as i32), BoundaryType::WallDirichlet, 0.01f32)); // Left wall
        // b_conds.push(BoundaryGroup::new(Rect::new_all(SIZE_X as i32 - 1, 0, 1, SIZE_Y as i32), BoundaryType::WallDirichlet, 0f32)); // Right wall
        //
        // b_conds.push(BoundaryGroup::new(Rect::new_all((SIZE_X / 3) as i32, 0, 1, (3 * SIZE_Y / 4) as i32), BoundaryType::WallDirichlet, 0f32));
        // b_conds.push(BoundaryGroup::new(Rect::new_all((2 * SIZE_X / 3) as i32, (1 * SIZE_Y / 4) as i32, 1, (3 * SIZE_Y / 4) as i32), BoundaryType::WallDirichlet, 0f32));


        // Vertical flow
        b_conds.push(BoundaryGroup::new(Rect::new_all(0i32, 0i32, SIZE_X as i32, 1i32), BoundaryType::WallDirichlet, 0.03f32)); // Top wall
        b_conds.push(BoundaryGroup::new(Rect::new_all(0, SIZE_Y as i32 - 1, SIZE_X as i32, 1), BoundaryType::WallOpen, 0.97f32)); // Bottom wall
        b_conds.push(BoundaryGroup::new(Rect::new_all(0, 0, 1, SIZE_Y as i32), BoundaryType::WallSymmetric, 0f32)); // Left wall
        b_conds.push(BoundaryGroup::new(Rect::new_all(SIZE_X as i32 - 1, 0, 1, SIZE_Y as i32), BoundaryType::WallSymmetric, 0f32)); // Right wall

        b_conds.push(BoundaryGroup::new(Rect::new_all((1 * SIZE_X / 4) as i32, (SIZE_Y / 5) as i32, (3 * SIZE_X / 4) as i32, 1), BoundaryType::WallSymmetric, 0f32));
        b_conds.push(BoundaryGroup::new(Rect::new_all((0 * SIZE_X / 4) as i32, (2 * SIZE_Y / 5) as i32, (3 * SIZE_X / 4) as i32, 1), BoundaryType::WallSymmetric, 0f32));
        b_conds.push(BoundaryGroup::new(Rect::new_all(0, (3 * SIZE_Y / 5) as i32, (3 * SIZE_X / 4) as i32, 1), BoundaryType::WallSymmetric, 0f32));
        b_conds.push(BoundaryGroup::new(Rect::new_all((1 * SIZE_X / 4) as i32, (4 * SIZE_Y / 5) as i32, (3 * SIZE_X / 4) as i32, 1), BoundaryType::WallSymmetric, 0f32));
        b_conds.push(BoundaryGroup::new(Rect::new_all((5 * SIZE_X / 6) as i32, (1 * SIZE_Y / 2) as i32, (1 * SIZE_X / 10) as i32, 1), BoundaryType::WallSymmetric, 0f32));

        b_conds.push(BoundaryGroup::new(Rect::new_all(1, (4 * SIZE_Y / 9) as i32, 1, (1 * SIZE_Y / 8) as i32), BoundaryType::WallOpen, 0.92f32)); // Left wall

        // Vertical turbulent test
        // b_conds.push(BoundaryGroup::new(Rect::new_all(0i32, 0i32, SIZE_X as i32, 1i32), BoundaryType::WallOpen, 1.0f32)); // Top wall
        // b_conds.push(BoundaryGroup::new(Rect::new_all((3 * SIZE_X / 7) as i32, 2, (1 * SIZE_X / 7) as i32, 1), BoundaryType::WallDirichlet, 0.3f32)); // Top wall emitter
        // b_conds.push(BoundaryGroup::new(Rect::new_all(0, SIZE_Y as i32 - 1, SIZE_X as i32, 1), BoundaryType::WallOpen, 0.9f32)); // Bottom wall
        // b_conds.push(BoundaryGroup::new(Rect::new_all(0, 0, 1, SIZE_Y as i32), BoundaryType::WallSymmetric, 0f32)); // Left wall
        // b_conds.push(BoundaryGroup::new(Rect::new_all(SIZE_X as i32 - 1, 0, 1, SIZE_Y as i32), BoundaryType::WallSymmetric, 0f32)); // Right wall

        b_conds.push(BoundaryGroup::new(Rect::new_all((3 * SIZE_X / 7) as i32, (1 * SIZE_Y / 2) as i32, (1 * SIZE_X / 7) as i32, 1), BoundaryType::WallSymmetric, 0f32));

        for b_cond in b_conds.iter() {
            // Loop over the entire area of the boundary group

            // Set the cells for all boundary condition to type wall
            for x in b_cond.rect.min.0..b_cond.rect.max.0 {
                for y in b_cond.rect.min.1..b_cond.rect.max.1 {
                    let (x, y) = ((x as usize), (y as usize));
                    sim[x][y].cell_type = CellType::CellTypeWall;
                }
            }
        }

        for x in 0..sim.len() {
            for y in 0..sim[0].len() {
                if sim[x][y].cell_type == CellType::CellTypeWall {
                    continue;
                }
                if x < SIZE_X / 3 {
                    sim[x][y].density = 1.0;
                } else {
                    sim[x][y].density = MIN_DENSITY_VAL;
                }
            }
        }

        for x in 0..sim.len() {
            for y in 0..sim[0].len() {
                let mut cell = &mut sim[x][y];
                if cell.cell_type == CellType::CellTypeWall {
                    continue;
                }
                for (i, &e) in NEIGHBORS.iter().enumerate() {

                    cell.eq_fn[i] = WEIGHTS[i] * cell.density;
                    cell.in_fn[i] = cell.eq_fn[i];
                }
            }
        }

        Self {
            sim,
            boundary_conditions: b_conds,
            sim_step: 0,
            last_sim_step: Instant::now(),
            img_texture: None,
            magnification: (800 / SIZE_Y) as f32,
            velocity_view: false,
            show_trajectories: false,
            is_playing: false,
        }
    }
}

// Neighbor direction mapping
#[derive(Clone, Copy)]
pub enum BC_Direction {
    CENTER = 0,
    East = 1,
    West = 2,
    South = 3,
    North = 4,
    SE = 5,
    NW = 6,
    NE = 7,
    SW = 8,
}

#[derive(Clone, Copy)]
pub enum Side {
    North = 4,
    South = 3,
    East = 1,
    West = 2,
}

// Returns outward normal unit vector for each side:
fn normal_out(side: Side) -> (i32, i32) {
    match side {
        Side::North => (0, -1),
        Side::South => (0, 1),
        Side::East  => (1, 0),
        Side::West  => (-1, 0),
    }
}

// Checks if direction i is unknown (would stream from outside) given outward normal:
fn is_unknown(i: usize, n_out: (i32,i32)) -> bool {
    let (cx, cy) = NEIGHBORS[i];
    (cx * n_out.0 + cy * n_out.1) > 0
}

fn apply_velocity_bc(
    state: &mut Vec<Vec<CellInfo>>,
    x: usize,
    y: usize,
    side: Side,
    b_cond: &BoundaryGroup,
) {
    let (u_x, u_y) = (0f32, b_cond.boundary_value);
    let (nx, ny) = normal_out(side);
    let u_dot_n = u_x *(nx as f32) + u_y *(ny as f32);

    let cell = &state[x][y];
    // sum_known
    let sum_known = (0..9)
        .into_iter()
        .filter(|&i| !is_unknown(i, (nx, ny)))
        .fold(0.0f32, |acc, i| acc + cell.in_fn[i]);

    let opposite_mapping = [0, 2, 1, 4, 3, 6, 5, 8, 7];
    let rho = sum_known / (1.0f32 - u_dot_n);

    let cell = &mut state[x][y];
    for i in 0..9 {
        if is_unknown(i, (nx, ny)) {
            let opp = opposite_mapping[i];
            let (cx, cy) = NEIGHBORS[i];
            if cx == 0 || cy == 0 {
                // axis direction: this should be the index along normal
                // f[i] = f[opp] + 2/3 * rho * u_n
                cell.in_fn[i] = cell.in_fn[opp] + 2.0/3.0 * rho * u_dot_n;
            } else {
                // diagonal
                let c_dot_u = (cx as f32)* u_x + (cy as f32)* u_y;
                cell.in_fn[i] = cell.in_fn[opp] + (1.0/6.0)*rho*c_dot_u;
            }
        }
    }

}

fn apply_open_bc(
    state: &mut Vec<Vec<CellInfo>>,
    x: usize,
    y: usize,
    side: Side,
    b_cond: &BoundaryGroup,
) {
    // Prescribed density ρ0:
    let rho0 = b_cond.boundary_value; // Assume boundary_value holds the density for WallOpen
    // After streaming, the known distributions are in state[x][y].in_fn.
    // Sum known f_i:
    let (nx, ny) = normal_out(side);

    let cell = &state[x][y];
    let sum_known = (0..9)
        .into_iter()
        .filter(|&i| !is_unknown(i, (nx, ny)))
        .fold(0.0f32, |acc, i| acc + cell.in_fn[i]);
    // Compute normal velocity component u_n:
    // For the side, derive formula analogous to west example. We implement generic:
    // Let A = sum_known.
    // Let sum_mom_known = sum_{i in known} c_{i,n} * f_i.

    let sum_mom_known = NEIGHBORS.iter().enumerate()
        .fold(0.0f32, |acc, (i, c)| {
            if !is_unknown(i, (nx, ny)) {
                // normal component c_{i,n} = dot( (cx,cy), normal basis ). But for unit normals aligned with axes,
                // c_{i,n} = cx if side=East/West, or cy if side=North/South.
                // Since normal_out gives (nx,ny) = ±(1,0) or ±(0,1), dot = cx*nx + cy*ny.
                acc + (c.0 as f32)* (nx as f32) * cell.in_fn[i]  // equals c_{i,n} * f_i since nx or ny is 0/±1
                    + (c.1 as f32)* (ny as f32) * cell.in_fn[i]
            }
            else {
                acc
            }
        });

    
    // Unknown sum Σ_{i in unknown} f_i = rho0 - A.
    // And Σ_{i in unknown} c_{i,n} f_i = (rho0 - A) * (+1), because for unknown i, c_{i,n}=+1 in normal coordinate (pointing into domain).
    // So total ρ0 * u_n = sum_mom_known + (rho0 - A).
    let u_n = (sum_mom_known + (rho0 - sum_known)) / rho0;
    // Decide tangential component u_t:
    // For simplicity, assume zero tangential velocity:
    let u_t = 0.0;
    // Construct full velocity vector (u_x,u_y) depending on side:
    let (u_x, u_y) = match side {
        Side::West  => (u_n, u_t),
        Side::East  => (-u_n, u_t),
        Side::South => (u_t, -u_n),
        Side::North => (u_t, u_n),
    };
    // Note: verify sign conventions carefully: above formula assumed for West boundary normal is (-1,0) so c_{i,n}=+1 for unknown dirs.

    let opposite_mapping = [0, 2, 1, 4, 3, 6, 5, 8, 7];

    // Zou–He reconstruction:
    {
        let cell = &mut state[x][y];
        for i in 0..9 {
            if is_unknown(i, (nx, ny)) {
                let opp = opposite_mapping[i];
                let (cx, cy) = NEIGHBORS[i];
                // Check if axis direction (|c|=1 and aligned with normal):
                if (cx == 0 && cy == 0) {
                    // Should not happen for unknown (f0 is known).
                    continue;
                }
                if cx == 0 || cy == 0 {
                    // axis: c aligned with x or y => this is the normal direction index
                    cell.in_fn[i] = cell.in_fn[opp] + (2.0/3.0) * rho0 * u_n;
                } else {
                    // diagonal: f[i] = f[opp] + 1/6 * rho0 * (c_x * u_x + c_y * u_y)
                    let c_dot_u = (cx as f32)*u_x + (cy as f32)*u_y;
                    cell.in_fn[i] = cell.in_fn[opp] + (1.0/6.0) * rho0 * c_dot_u;
                }
            }
        }
    }
}

pub fn step_sim(mut state: &mut Vec<Vec<CellInfo>>, b_conds: &Vec<BoundaryGroup>, sim_step: &mut usize) {
    //let now = Instant::now();
    //let mut new_state: Vec<Vec<CellInfo>> = vec![vec![CellInfo::default(); N]; N];

    let (n_x, n_y) = (state.len(), state[0].len());

    let mut gas_particles = 0f32;
    let mut full_cells = 0;

    let dt = 1f32;
    let tau = 1f32;

    //let NEIGHBORS = [(1, 0), (0, -1i32), (-1i32, 0), (0, 1)];
    // let NEIGHBORS = [
    //     (-1,1),(0,1),(1,1),(-1,0),(0,0),(1,0),(-1,-1),(0,-1),(1,-1),
    // ];
    // let NEIGHBORS = [
    //     (0,0), (1,0), (-1,0), (0,1), (0, -1), (1,1), (-1,1), (-1,-1), (1,-1)
    // ];
    //let NEIGHBORS = [(1, 0), (-1i32, 0), (0, 1), (0, -1i32)];

    // Collision
    for x in 1..n_x - 1 {
        for y in 1..n_y - 1 {
            let cell = &mut state[x][y];
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
            let mut u = NEIGHBORS
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
            for (i, &e) in NEIGHBORS.iter().enumerate() {
                let (n_x, n_y) = (e.0 as f32, e.1 as f32);
                let eu = (n_x * u.0 + n_y * u.1);
                cell.eq_fn[i] =
                    WEIGHTS[i] * cell.density * (1.0 + 3.0 * eu + 4.5 * eu.powi(2) - u_mag);
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
            for (i, &e) in NEIGHBORS.iter().enumerate() {
                let (new_x, new_y): (_, _) = ((x as i32 - e.0) as usize, (y as i32 - e.1) as usize);

                if state[new_x][new_y].cell_type == CellType::CellTypeWall {

                    let side = if i == BC_Direction::West as usize {
                        Side::West
                    } else if i == BC_Direction::East as usize {
                        Side::East
                    } else if i == BC_Direction::South as usize {
                        Side::South
                    } else if i == BC_Direction::North as usize {
                        Side::North
                    } else {
                        continue; // only handle actual boundary nodes
                    };

                    for b_cond in b_conds {
                        if b_cond.rect.contains((new_x as i32, new_y as i32)) {
                            match b_cond.boundary_type {
                                BoundaryType::WallBounceBack => {
                                    // if e.0 == 1 || (e.1 == 1 && e.0 == 0) {
                                    if i % 2 == 1 {
                                        state[x][y].in_fn[i] = state[x][y].out_fn[(i + 1)];
                                    } else {
                                        state[x][y].in_fn[i] = state[x][y].out_fn[(i - 1)];
                                    }
                                }
                                BoundaryType::WallSymmetric => {
                                    //let symmetric_mapping = [0,2,1,4,3,8,7,6,5];
                                    let symmetric_mapping = [0, 2, 1, 4, 3, 7, 8, 5, 6];
                                    state[x][y].in_fn[i] = state[x][y].out_fn[symmetric_mapping[i]];
                                }
                                BoundaryType::WallDirichlet => {
                                    apply_velocity_bc(&mut state, x, y, side, b_cond);
                                }
                                BoundaryType::WallOpen => {

                                    apply_open_bc(&mut state, x, y, side, b_cond);
                                }
                            }
                        }
                    }

                    continue;
                }

                state[x][y].in_fn[i] = state[new_x][new_y].out_fn[i];
            }
        }
    }

    println!("elements: {gas_particles}, gas_cells: {full_cells}");

    *sim_step += 1;
    //*state.clone()
}
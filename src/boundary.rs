

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BoundaryType {
    WallBounceBack,
    WallSymmetric,
    WallDirichlet,
    WallOpen,
}

impl TryFrom<u8> for BoundaryType {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == BoundaryType::WallBounceBack as u8 => Ok(BoundaryType::WallBounceBack),
            x if x == BoundaryType::WallSymmetric as u8 => Ok(BoundaryType::WallSymmetric),
            x if x == BoundaryType::WallDirichlet as u8 => Ok(BoundaryType::WallDirichlet),
            x if x == BoundaryType::WallOpen as u8 => Ok(BoundaryType::WallOpen),
            _ => Err(()),
        }
    }
}

pub struct Rect {
    pub min: (i32, i32),
    pub max: (i32, i32),
}

impl Rect {
    pub fn new(min: (i32, i32), max: (i32, i32)) -> Self {
        Self { min, max }
    }
    
    pub fn new_all(m_x: i32, m_y: i32, w: i32, h: i32) -> Self {
        Self::new((m_x, m_y), (m_x + w, m_y + h))
    }

    pub fn contains(&self, p: (i32, i32)) -> bool {
        p.0 >= self.min.0 && p.0 < self.max.0 && p.1 >= self.min.1 && p.1 < self.max.1
    }
}

pub struct BoundaryGroup {
    pub rect: Rect,

    pub boundary_type: BoundaryType,
    pub boundary_value: f32,                // Constant value of the boundary condition (eg. 0 density)
    pub boundary_flow_func: Option<fn(f32) -> f32>, // Function of flow. boundary_value should be NAN when using this.
}

impl BoundaryGroup {
    pub fn new(rect: Rect, boundary_type: BoundaryType, boundary_value: f32) -> Self {
        Self { rect, boundary_type, boundary_value, boundary_flow_func: None }
    }

    pub fn new_func(rect: Rect, boundary_type: BoundaryType, boundary_value: f32, boundary_flow_func: fn(f32) -> f32) -> Self {
        Self { rect, boundary_type, boundary_value, boundary_flow_func: Some(boundary_flow_func) }
    }
}
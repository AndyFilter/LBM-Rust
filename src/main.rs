mod gui;
mod sim;
mod boundary;

use crate::sim::step_sim;
use crate::{sim::CellInfo, sim::CellType};
use crate::boundary::{BoundaryGroup, BoundaryType};
use eframe::{egui, App, NativeOptions};
use egui::Vec2;
use image::GenericImage;
use rand::Rng;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io;
use std::io::{BufRead, BufWriter, Write};
use std::time::Instant;

struct MyApp {
    sim: Vec<Vec<CellInfo>>, // NxN grid with states
    boundary_conditions: Vec<BoundaryGroup>,
    sim_step: usize,
    last_sim_step: Instant,

    img_texture: Option<egui::TextureHandle>,
    magnification: f32,

    velocity_view: bool,
    show_trajectories: bool,
    is_playing: bool,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }
}

impl MyApp {
    fn save_sim_to_file(&self, file_name: String) -> Result<(), io::Error> {
        let file = File::create(file_name + ".sim")?;
        // Buffered writing for better performance
        let mut writer = BufWriter::new(file);

        write!(writer, "{};{};{}\n", self.sim.len(), self.sim[0].len(), self.sim_step)?;
        for x in 0..self.sim.len() {
            for y in 0..self.sim[0].len() {
                writeln!(writer, "{}", self.sim[x][y])?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    fn import_sim_from_file(&mut self, file_name: String) -> Result<(), io::Error> {
        let file = File::open(file_name + ".sim")?;
        let mut reader = io::BufReader::new(file);
        let (size_x, size_y, step): (usize, usize, usize);

        self.sim.clear();

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

        let mut first_line = Default::default();
        reader.read_line(&mut first_line)?;
        first_line = first_line.trim().parse().unwrap(); // remove the \n character
        let parts: Vec<&str> = first_line.split(';').collect();
        println!("{:?}", parts);
        size_x = parts[0].parse().unwrap();
        size_y = parts[1].parse().unwrap();
        step = parts[2].parse().unwrap();

        println!("Imported sim size: {}x{}, at step {}", size_x, size_y, step);

        self.sim = vec![vec![CellInfo::default(); size_y]; size_x];

        self.sim_step = step;

        let mut cell: CellInfo = Default::default();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;

            // Split the line into the array part and the values part
            let parts: Vec<&str> = line.split(';').collect();
            if parts.len() < 2 {
                eprintln!("Invalid line format: {}", line);
                continue;
            }

            // f_in
            // Parse the array part
            let array_part = parts[0].trim_start_matches('[').trim_end_matches(']');
            cell.in_fn = <[f32; 9]>::try_from(array_part
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect::<Vec<f32>>()).unwrap();
            cell.out_fn = cell.in_fn;

            // cell.density = 1.0;
            cell.cell_type = CellType::try_from(parts[1].trim().parse::<u8>().unwrap()).unwrap();
            //cell.boundary_type = BoundaryType::try_from(parts[2].trim().parse::<u8>().unwrap()).unwrap();

            // Recalculate rest of the parameters for this cell
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

            self.sim[i/size_x][i%size_y] = cell;
        }

        self.sim_step -= 1; // account for the step that does nothing
        step_sim(&mut self.sim, &self.boundary_conditions, &mut self.sim_step); // "fake" step used only to recalculate values
        Ok(())
    }
}

fn main() {
    let mut native_options = NativeOptions::default();

    native_options.viewport.inner_size = Some(Vec2::new(1400f32, 850f32));
    eframe::run_native(
        "LBM-Rust",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(MyApp::new(cc)))
        }),
    )
    .expect("Could not start the window");
}

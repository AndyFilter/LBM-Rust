mod gui;
mod sim;

use crate::sim::{step_sim, WallType, PLOT_COLORS_U8};
use crate::{sim::CellInfo, sim::CellType, sim::MIN_DENSITY_VAL};
use eframe::egui::TextureOptions;
use eframe::{egui, epaint, App, NativeOptions};
use egui::Vec2;
use image::{GenericImage, ImageBuffer, Rgba};
use rand::Rng;
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::time::Instant;
use std::io;

struct MyApp {
    sim: Vec<Vec<CellInfo>>, // NxN grid with states
    // sim_state_change: Vec<Vec<u32>>,
    sim_step: usize,
    last_sim_step: Instant,

    img_texture: Option<egui::TextureHandle>,
    magnification: f32,

    velocity_view: bool,
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
            cell.wall_type = WallType::try_from(parts[2].trim().parse::<u8>().unwrap()).unwrap();

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
        step_sim(&mut self.sim, &mut self.sim_step); // "fake" step used only to recalculate values
        Ok(())
    }
}

fn map_range(from_range: (f32, f32), to_range: (f32, f32), s: f32) -> f32 {
    to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
}

fn create_image_from_buffer(
    buffer: &Vec<Vec<CellInfo>>,
    width: u32,
    height: u32,
    ctx: &egui::Context,
    velocity_view: bool,
    velocity_axis: usize,
    mut opt_image_buffer: Option<&mut ImageBuffer<Rgba<u8>, Vec<u8>>>,
) -> egui::TextureHandle {
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let velo_dimness = 2.0f32;

    for (x, row) in buffer.iter().enumerate() {
        for (y, &ref pixel) in row.iter().enumerate() {
            let mut col = Rgba(PLOT_COLORS_U8[buffer[x][y].cell_type as usize]);
            //if buffer[x][y].cell_type == CellTypeGas {
            if velocity_view && pixel.cell_type != CellType::CellTypeWall {
                if buffer[x][y].velocity[velocity_axis] >= 0.0 {
                    col.0[0] = (256f32
                        * map_range(
                            (
                                0.000 * velo_dimness / MIN_DENSITY_VAL,
                                0.01 * velo_dimness / MIN_DENSITY_VAL,
                            ),
                            (0.01, 1.0),
                            buffer[x][y].velocity[velocity_axis],
                        ))
                    .floor() as u8;
                } else {
                    col.0[2] = (256f32
                        * map_range(
                            (
                                0.000 * velo_dimness / MIN_DENSITY_VAL,
                                0.01 * velo_dimness / MIN_DENSITY_VAL,
                            ),
                            (0.01, 1.0),
                            -buffer[x][y].velocity[velocity_axis],
                        ))
                    .floor() as u8;
                }
            } else if (256f32 * buffer[x][y].density).floor() as u8 > 0 {
                // col.0[1] = 32;
                // col.0[2] = 32;
                //col.0[0] = (64f32 * buffer[x][y].density) as u8
                col.0.fill(
                    (256f32
                        * map_range(
                            (MIN_DENSITY_VAL - 0.08, MIN_DENSITY_VAL + 0.1),
                            (0.1, 1.0),
                            buffer[x][y].density,
                        ))
                    .floor() as u8,
                );
            }
            col.0[3] = 255;
            img.put_pixel(x as u32, y as u32, col);
        }
    }
    if opt_image_buffer.is_some() {
        let buffer = opt_image_buffer.unwrap();
        //buffer = img.clone();
        img.clone_into(buffer);
        //opt_image_buffer.unwrap().copy_from(&img.clone(), img.width(), img.height()).expect("Could not copy image");
        //opt_image_buffer.insert(&mut img.clone());
    }

    // Convert the ImageBuffer to a ColorImage for egui
    let (width, height) = img.dimensions();
    let pixels = img.into_raw();
    let color_image =
        egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &pixels);

    let mut img_data = TextureOptions::default();
    img_data.magnification = epaint::textures::TextureFilter::Nearest;
    img_data.minification = epaint::textures::TextureFilter::Nearest;

    // Use the context passed from the creation function
    ctx.load_texture("sim_tex", color_image, img_data)
}

fn main() {
    let mut native_options = NativeOptions::default();

    native_options.viewport.inner_size = Some(Vec2::new(2450f32, 850f32));
    eframe::run_native(
        "LBM-Rust",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(MyApp::new(cc)))
        }),
    )
    .expect("Could not start the window");
}

use crate::{sim::CellInfo, step_sim, MyApp};
use eframe::{egui, epaint};
use eframe::egui::{Color32, Painter, Pos2, TextureOptions, Vec2};
use image::{ImageBuffer, Rgba};
use std::time::{Duration, Instant};
use crate::sim::{CellType, MIN_DENSITY_VAL, PLOT_COLORS_U8};

fn draw_local_velocities(lines_offset: usize, sim: &Vec<Vec<CellInfo>>, painter: &Painter,
                         color: Option<Color32>, pos_offset: Pos2, scale: f32) {
    for x in (0..sim.len()).step_by(lines_offset) {
        for y in (0..sim[0].len()).step_by(lines_offset) {
            let velo = Vec2::new(
                sim[x][y].velocity[0],
                sim[x][y].velocity[1],
            ) * 100.0; // 100 - length
            let start = Vec2::new(x as f32, y as f32);
            let end = Vec2::new(x as f32 + velo.x, y as f32 + velo.y);
            painter.line_segment(
                [
                    pos_offset + start * scale,
                    pos_offset + end * scale,
                ],
                (1.0, color.unwrap_or(Color32::WHITE)), // Line width and color
            );
        }
    }
}

fn calc_trac_points(lines_count: usize, mu: f32, sim: &Vec<Vec<CellInfo>>) -> Vec<Vec<Pos2>> {
    let sim_size: (usize, usize) = (sim.len(), sim[0].len());
    let grav = -0.0005f32;
    let mut global_positions: Vec<Vec<Pos2>> = Vec::new();
    for i in 0..lines_count {
        global_positions.push(Vec::with_capacity(sim_size.1 / 10));
        //let mut positions: &mut std::vec::Vec<Pos2> = global_positions.last().unwrap();//Vec::with_capacity(sim_size.1 / 10); // = Vec::from(Vec2::new(0.0, (i * (sim_size.1 - 4)) as f32 / (lines_count) as f32));
        //let mut pos: Pos2 = Pos2::new(2.0, ((i * (sim_size.1 - 4)) as f32 / (lines_count - 1) as f32) + 2.0); //  + ((sim_size.1) / lines_count / 2) as f32 // On the X axis
        let mut pos: Pos2 = Pos2::new(((i * (sim_size.0 - 4)) as f32 / (lines_count - 1) as f32) + 2.0, 0f32); //  + ((sim_size.1) / lines_count / 2) as f32 // On the Y axis
        let mut velocity: Vec2 = Vec2::new(sim[pos.x as usize][pos.y as usize].velocity[0], sim[pos.x as usize][pos.y as usize].velocity[1]);
        let mut iter = 0;
        while pos.x >= 0f32 && pos.x < (sim_size.0 as f32 - 1.0) && pos.y >= 0f32 && pos.y < sim_size.1 as f32 && iter < 999 {
            global_positions[i].push(pos);
            let mut new_vel: Vec2 = Vec2::new(
                mu * velocity.x + (1.0 - mu) * sim[pos.x as usize][pos.y as usize].velocity[0],
                mu * velocity.y + (1.0 - mu) * sim[pos.x as usize][pos.y as usize].velocity[1] - grav
            );

            let new_pos = pos + 100.0 * (new_vel + velocity) / 2.0;
            velocity = new_vel;
            pos = new_pos;
            iter += 1;
        }
        //global_positions.push(positions);
    }

    global_positions
}

fn draw_traj_lines(trajectories: &Vec<Vec<Pos2>>, painter: &Painter, color: Option<Color32>, pos_offset: Pos2, scale: f32) {
    for trajectory in trajectories {
        painter.line(trajectory
                         .into_iter()
                         .map(|&a| pos_offset + a.to_vec2() * scale)
                         .collect::<Vec<_>>(), (1.0, color.unwrap_or(Color32::BLACK))
        );
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                // GUI controls

                let mut save_image = false;
                ui.horizontal(|ui| {
                    let now = Instant::now();
                    if ui.button("Step").clicked()
                        || (self.is_playing
                        && (now - self.last_sim_step) > Duration::from_millis(5))
                    {
                        //self.sim.push(step_sim(self.sim.last().unwrap()));
                        //self.sim = step_sim(&mut self.sim, &mut self.sim_step);
                        step_sim(&mut self.sim, &self.boundary_conditions, &mut self.sim_step);
                        self.last_sim_step = now;
                        let elapsed = now.elapsed();
                        println!("Stepping sim took: {:.2?}", elapsed);
                        //sleep(time::Duration::from_millis(25));
                        ctx.request_repaint_after(Duration::from_millis(5));
                    }

                    let b = ui.button(if self.is_playing { "Stop" } else { "Start" });
                    if b.clicked() {
                        self.is_playing = !self.is_playing;
                    }

                    //ui.checkbox(&mut self.velocity_view, "Velocity view");

                    ui.label(format!("step: {}", self.sim_step));

                    save_image = ui.button("Save Image").clicked();
                    if ui.button("Export Sim").clicked() {
                        if self.save_sim_to_file("sim_save".to_string()).is_err() {
                            println!("Export Sim to file failed");
                        }
                    }

                    if ui.button("Import Sim").clicked() {
                        if self.import_sim_from_file("sim_save".to_string()).is_err() {
                            println!("Import Sim from file failed");
                        }
                    }

                    ui.checkbox(&mut self.show_trajectories, "Show trajectories");
                });

                let mut img_buff: ImageBuffer<Rgba<u8>, Vec<u8>> = Default::default();
                //let mut pixels = vec![0u8; (self.sim.len() * self.sim[0].len() * 4) as usize]; // RGBA format

                let lines_count = if self.show_trajectories { 8 } else { 0 };
                let low_mass_traj = calc_trac_points(lines_count, 0.05, &self.sim);
                let mid_mass_traj = calc_trac_points(lines_count, 0.8, &self.sim);
                let high_mass_traj = calc_trac_points(lines_count, 0.9, &self.sim);

                //let now = Instant::now();
                self.img_texture = Some(create_image_from_buffer(
                    &self.sim,
                    self.sim.len() as u32,
                    self.sim[0].len() as u32,
                    ctx,
                    false,
                    0,
                    if save_image {
                        Some(&mut img_buff)
                    } else {
                        None
                    },
                ));

                if save_image {
                    img_buff
                        .save(format!("Sim_Dens_Step_{}.bmp", self.sim_step))
                        .expect("Failed to save image");
                }

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        if let Some(texture) = &self.img_texture {
                            let mut tx = egui::Image::new(texture);
                            tx = tx.fit_to_exact_size(Vec2::new(
                                self.sim.len() as f32 * self.magnification,
                                self.sim[0].len() as f32 * self.magnification,
                            ));
                            let image_resp = ui.add(tx);

                            //let image_resp = ui.image(texture);

                            if image_resp.contains_pointer() {
                                let mut local_pos;
                                if let Some(pp) = ctx.pointer_latest_pos() {
                                    local_pos = pp - image_resp.interact_rect.min;
                                    local_pos = Vec2::new(
                                        (local_pos.x + 0.5).floor(),
                                        (local_pos.y + 0.5).floor(),
                                    ) / self.magnification;

                                    if (local_pos.x < self.sim.len() as f32
                                        && local_pos.y < self.sim[0].len() as f32)
                                        && (local_pos.x > 0f32 && local_pos.y > 0f32)
                                    {
                                        let density = self.sim[local_pos.x as usize]
                                            [local_pos.y as usize]
                                            .density;
                                        ui.vertical(|ui| {
                                            ui.label(format!(
                                                "density: {} ({}, {})",
                                                density, local_pos.x as usize, local_pos.y as usize
                                            ));
                                        });
                                    }
                                } else {
                                    local_pos = Vec2::ZERO;
                                }
                            }

                            // Get the painter to draw over the image
                            let painter = ui.painter();

                            let image_pos = image_resp.rect.min; // Get the position of the image
                            let image_scale = self.magnification; // Scale factor

                            draw_local_velocities(10, &self.sim, painter, Option::from(Color32::BLACK), image_pos, image_scale);

                            if self.show_trajectories {
                                draw_traj_lines(&low_mass_traj, painter, Option::from(Color32::LIGHT_BLUE), image_pos, image_scale);
                                draw_traj_lines(&mid_mass_traj, painter, Option::from(Color32::GREEN), image_pos, image_scale);
                                draw_traj_lines(&high_mass_traj, painter, Option::from(Color32::RED), image_pos, image_scale);
                            }
                        }
                    });

                    self.img_texture = Some(create_image_from_buffer(
                        &self.sim,
                        self.sim.len() as u32,
                        self.sim[0].len() as u32,
                        ctx,
                        true,
                        0,
                        if save_image {
                            Some(&mut img_buff)
                        } else {
                            None
                        },
                    ));

                    if save_image {
                        img_buff
                            .save(format!("Sim_Vx_Step_{}.bmp", self.sim_step))
                            .expect("Failed to save image");
                    }

                    ui.vertical(|ui| {
                        if let Some(texture) = &self.img_texture {
                            let mut tx = egui::Image::new(texture);
                            tx = tx.fit_to_exact_size(Vec2::new(
                                self.sim.len() as f32 * self.magnification,
                                self.sim[0].len() as f32 * self.magnification,
                            ));
                            let image_resp = ui.add(tx);

                            //let image_resp = ui.image(texture);

                            if image_resp.contains_pointer() {
                                let mut local_pos;
                                if let Some(pp) = ctx.pointer_latest_pos() {
                                    local_pos = pp - image_resp.interact_rect.min;
                                    local_pos = Vec2::new(
                                        (local_pos.x + 0.5).floor(),
                                        (local_pos.y + 0.5).floor(),
                                    ) / self.magnification;

                                    if (local_pos.x < self.sim.len() as f32
                                        && local_pos.y < self.sim[0].len() as f32)
                                        && (local_pos.x > 0f32 && local_pos.y > 0f32)
                                    {
                                        let density = self.sim[local_pos.x as usize]
                                            [local_pos.y as usize]
                                            .velocity[0];
                                        ui.label(format!(
                                            "vx: {} ({}, {})",
                                            density, local_pos.x as usize, local_pos.y as usize
                                        ));
                                    }
                                } else {
                                    local_pos = Vec2::ZERO;
                                }
                            }

                            // Get the painter to draw over the image
                            let painter = ui.painter();

                            let image_pos = image_resp.rect.min; // Get the position of the image
                            let image_scale = self.magnification; // Scale factor

                            draw_local_velocities(10, &self.sim, painter, Option::from(Color32::WHITE), image_pos, image_scale);

                            if self.show_trajectories {
                                draw_traj_lines(&low_mass_traj, painter, Option::from(Color32::LIGHT_BLUE), image_pos, image_scale);
                                draw_traj_lines(&mid_mass_traj, painter, Option::from(Color32::GREEN), image_pos, image_scale);
                                draw_traj_lines(&high_mass_traj, painter, Option::from(Color32::RED), image_pos, image_scale);
                            }
                        }
                    });

                    self.img_texture = Some(create_image_from_buffer(
                        &self.sim,
                        self.sim.len() as u32,
                        self.sim[0].len() as u32,
                        ctx,
                        true,
                        1,
                        if save_image {
                            Some(&mut img_buff)
                        } else {
                            None
                        },
                    ));

                    if save_image {
                        img_buff
                            .save(format!("Sim_Vy_Step_{}.bmp", self.sim_step))
                            .expect("Failed to save image");
                    }

                    ui.vertical(|ui| {
                        if let Some(texture) = &self.img_texture {
                            let mut tx = egui::Image::new(texture);
                            tx = tx.fit_to_exact_size(Vec2::new(
                                self.sim.len() as f32 * self.magnification,
                                self.sim[0].len() as f32 * self.magnification,
                            ));
                            let image_resp = ui.add(tx);

                            //let image_resp = ui.image(texture);

                            if image_resp.contains_pointer() {
                                let mut local_pos;
                                if let Some(pp) = ctx.pointer_latest_pos() {
                                    local_pos = pp - image_resp.interact_rect.min;
                                    local_pos = Vec2::new(
                                        (local_pos.x + 0.5).floor(),
                                        (local_pos.y + 0.5).floor(),
                                    ) / self.magnification;

                                    if (local_pos.x < self.sim.len() as f32
                                        && local_pos.y < self.sim[0].len() as f32)
                                        && (local_pos.x > 0f32 && local_pos.y > 0f32)
                                    {
                                        let density = self.sim[local_pos.x as usize]
                                            [local_pos.y as usize]
                                            .velocity[1];
                                        ui.label(format!(
                                            "vy: {} ({}, {})",
                                            density, local_pos.x as usize, local_pos.y as usize
                                        ));
                                    }
                                } else {
                                    local_pos = Vec2::ZERO;
                                }
                            }

                            // Get the painter to draw over the image
                            let painter = ui.painter();

                            let image_pos = image_resp.rect.min; // Get the position of the image
                            let image_scale = self.magnification; // Scale factor

                            draw_local_velocities(10, &self.sim, painter, Option::from(Color32::WHITE), image_pos, image_scale);

                            if self.show_trajectories {
                                draw_traj_lines(&low_mass_traj, painter, Option::from(Color32::LIGHT_BLUE), image_pos, image_scale);
                                draw_traj_lines(&mid_mass_traj, painter, Option::from(Color32::GREEN), image_pos, image_scale);
                                draw_traj_lines(&high_mass_traj, painter, Option::from(Color32::RED), image_pos, image_scale);
                            }
                        }
                    });
                });

                // use egui::{RichText};
                // ui.label(RichText::new(format!("Spaleni ludzie: {}", self.casualties)).size(24.0));
            });
        });

        if self.is_playing {
            ctx.request_repaint_after(Instant::now() - self.last_sim_step);
        }
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
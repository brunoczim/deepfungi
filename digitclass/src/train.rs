use crate::{mnist, state, Network, Result};
use deepfungi::functions::{LogisticFn, SquaredError};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "train")]
#[structopt(about = "Trains a neural network")]
pub struct Command {
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: Option<PathBuf>,
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: Option<PathBuf>,
    #[structopt(short = "I", long = "images", parse(from_os_str))]
    images: PathBuf,
    #[structopt(short = "l", long = "labels", parse(from_os_str))]
    labels: PathBuf,
    #[structopt(short = "b", long = "begin")]
    begin: Option<u32>,
    #[structopt(short = "e", long = "end")]
    end: Option<u32>,
    #[structopt(default_value = "1", short = "E", long = "epochs")]
    epochs: u32,
    #[structopt(default_value = "1.0", short = "r", long = "rate")]
    rate: f64,
}

impl Command {
    pub fn run(&mut self) -> Result<()> {
        let loaded =
            mnist::load(&self.images, &self.labels, self.begin, self.end)?;

        let mut network = match &self.input {
            Some(path) => state::load(path)?,
            None => Network::new(
                LogisticFn,
                SquaredError,
                loaded.width * loaded.height,
                &[16, 16, 10],
            ),
        };

        let mut input =
            Box::<[f64]>::from(vec![0.0; loaded.width * loaded.height]);
        let mut output = Box::<[f64]>::from(vec![0.0; 10]);

        println!("Training...\n");

        for i in 0 .. self.epochs {
            println!("Epoch {}/{}", i + 1, self.epochs);

            for (i, img) in loaded.images.iter().enumerate() {
                for elem in &mut *output {
                    *elem = 0.0;
                }
                output[img.label as usize] = 1.0;

                for (elem, &pixel) in input.iter_mut().zip(img.pixels.iter()) {
                    *elem = pixel as f64 / 255.0;
                }

                let loss = network.train(&input, &output, self.rate);

                print!(
                    "\rIteration {:>5}/{:<5}: loss={:<15.13}",
                    i + 1,
                    loaded.images.len(),
                    loss
                );
            }

            println!();
        }

        println!("{:#?}", network);

        if let Some(output) = &self.output {
            state::save(output, &network)?;
        }

        Ok(())
    }
}

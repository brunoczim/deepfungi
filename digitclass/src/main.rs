mod mnist;

use std::path::PathBuf;
use structopt::StructOpt;

type Result<T> = ::std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    App::from_args().run();
}

#[derive(Debug, StructOpt)]
#[structopt(name = "digitclass")]
#[structopt(about = "Digit classifier")]
enum App {
    Train(TrainCommand),
}

impl App {
    fn run(&mut self) -> Result<()> {
        match self {
            App::Train(cmd) => cmd.run(),
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "train")]
#[structopt(about = "Trains a neural network")]
struct TrainCommand {
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: Option<PathBuf>,
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: Option<PathBuf>,
    #[structopt(short = "I", long = "images", parse(from_os_str))]
    images: PathBuf,
    #[structopt(short = "l", long = "labels", parse(from_os_str))]
    labels: PathBuf,
    #[structopt(short = "b", long = "begin")]
    begin: Option<usize>,
    #[structopt(short = "e", long = "end")]
    end: Option<usize>,
}

impl TrainCommand {
    fn run(&mut self) -> Result<()> {
        Ok(())
    }
}

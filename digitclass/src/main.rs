mod mnist;
mod state;
mod train;

use deepfungi::{
    functions::{LogisticFn, SquaredError},
    network,
};
use std::process;
use structopt::StructOpt;

type Result<T> = ::std::result::Result<T, Box<dyn std::error::Error>>;

type Network = network::Neural<LogisticFn, SquaredError>;

fn main() {
    if let Err(err) = App::from_args().run() {
        eprintln!("{}", err);
        process::exit(-1);
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "digitclass")]
#[structopt(about = "Digit classifier")]
enum App {
    Train(train::Command),
}

impl App {
    fn run(&mut self) -> Result<()> {
        match self {
            App::Train(cmd) => cmd.run(),
        }
    }
}

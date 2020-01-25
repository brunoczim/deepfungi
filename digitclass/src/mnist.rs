use super::Result;
use std::{
    error::Error,
    fmt,
    fs::File,
    io::Read,
    ops::RangeBounds,
    path::Path,
};

pub const WIDTH: usize = 28;
pub const HEIGHT: usize = 28;

pub const IMAGES_MAGIC: [u8; 4] = [0, 0, 8, 3];
pub const LABELS_MAGIC: [u8; 4] = [0, 0, 8, 1];

#[derive(Debug)]
pub struct ImagesMagicError;

impl fmt::Display for ImagesMagicError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("images' file magic number is wrong")
    }
}

impl Error for ImagesMagicError {}

#[derive(Debug)]
pub struct LabelsMagicError;

impl fmt::Display for LabelsMagicError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("labels' file magic number is wrong")
    }
}

impl Error for LabelsMagicError {}

#[derive(Debug)]
pub struct Image {
    pub pixels: [[u8; WIDTH]; HEIGHT],
}

pub fn load_images<P, R>(path: P, range: R) -> Result<Vec<Image>>
where
    P: AsRef<Path>,
    R: RangeBounds<usize>,
{
    let mut file = File::open(path)?;
    let mut int = [0; 4];

    file.read_exact(&mut int)?;
    if int != IMAGES_MAGIC {
        Err(ImagesMagicError)?;
    }

    file.read_exact(&mut int)?;

    let mut imgs = Vec::new();

    Ok(imgs)
}

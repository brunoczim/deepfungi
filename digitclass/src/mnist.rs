use crate::Result;
use std::{
    convert::TryInto,
    error::Error,
    fmt,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

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
pub struct RangeError;

impl fmt::Display for RangeError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("the given range is invalid")
    }
}

impl Error for RangeError {}

#[derive(Debug)]
pub struct IncompatibleFiles;

impl fmt::Display for IncompatibleFiles {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("the given label and image files are incompatible")
    }
}

impl Error for IncompatibleFiles {}

#[derive(Debug)]
pub struct NoSpace;

impl fmt::Display for NoSpace {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("no space to store the images")
    }
}

impl Error for NoSpace {}

#[derive(Debug)]
pub struct BadLabel;

impl fmt::Display for BadLabel {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("labels must be between 0 and 9")
    }
}

impl Error for BadLabel {}

#[derive(Debug)]
pub struct Image {
    pub pixels: Box<[u8]>,
    pub label: u8,
}

#[derive(Debug)]
pub struct Loaded {
    pub images: Box<[Image]>,
    pub width: usize,
    pub height: usize,
}

fn read_header<R1, R2>(
    mut images_file: R1,
    mut labels_file: R2,
) -> Result<(u32, usize, usize)>
where
    R1: Read,
    R2: Read,
{
    let mut int = [0; 4];

    images_file.read_exact(&mut int)?;
    if int != IMAGES_MAGIC {
        Err(ImagesMagicError)?;
    }
    labels_file.read_exact(&mut int)?;
    if int != LABELS_MAGIC {
        Err(ImagesMagicError)?;
    }

    images_file.read_exact(&mut int)?;
    let count = u32::from_be_bytes(int);
    labels_file.read_exact(&mut int)?;
    if count != u32::from_be_bytes(int) {
        Err(IncompatibleFiles)?;
    }

    images_file.read_exact(&mut int)?;
    let width = u32::from_be_bytes(int).try_into()?;
    images_file.read_exact(&mut int)?;
    let height = u32::from_be_bytes(int).try_into()?;

    Ok((count, width, height))
}

fn compute_total<R1, R2>(
    mut images_file: R1,
    mut labels_file: R2,
    count: u32,
    width: usize,
    height: usize,
    begin: Option<u32>,
    end: Option<u32>,
) -> Result<u32>
where
    R1: Read + Seek,
    R2: Read + Seek,
{
    Ok(match begin {
        Some(offset) if offset < count => {
            labels_file.seek(SeekFrom::Current(offset as u64 as i64))?;

            let mut jump = (offset as u64)
                .checked_mul(width as u64)
                .and_then(|val| val.checked_mul(height as u64))
                .ok_or(NoSpace)?;
            while jump > i64::max_value() as u64 {
                let step = i64::max_value();
                images_file.seek(SeekFrom::Current(step))?;
                jump -= step as u64;
            }
            images_file.seek(SeekFrom::Current(jump as i64))?;

            match end {
                Some(end) if end < offset => Err(RangeError)?,

                Some(end) => end - offset + 1,
                None => count,
            }
        },

        Some(_) => Err(RangeError)?,

        None => end.map_or(count, |end| end + 1),
    })
}

pub fn load<P1, P2>(
    images_path: P1,
    labels_path: P2,
    begin: Option<u32>,
    end: Option<u32>,
) -> Result<Loaded>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    let mut images_file = BufReader::new(File::open(images_path)?);
    let mut labels_file = BufReader::new(File::open(labels_path)?);

    let (count, width, height) =
        read_header(&mut images_file, &mut labels_file)?;
    let total = compute_total(
        &mut images_file,
        &mut labels_file,
        count,
        width,
        height,
        begin,
        end,
    )?;
    let total = total.try_into()?;

    let mut imgs = Vec::with_capacity(total);
    let img_size = width.checked_mul(height).ok_or(NoSpace)?;

    for _ in 0 .. total {
        let mut pixels = Vec::with_capacity(img_size);

        for _ in 0 .. img_size {
            let mut buf = [0];
            images_file.read_exact(&mut buf)?;
            pixels.push(buf[0]);
        }

        let mut buf = [0];
        labels_file.read_exact(&mut buf)?;
        if buf[0] >= 10 {
            Err(BadLabel)?;
        }
        imgs.push(Image { pixels: pixels.into(), label: buf[0] });
    }

    Ok(Loaded { images: imgs.into(), width, height })
}

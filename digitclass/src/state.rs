use crate::{Network, Result};
use std::{fs::File, path::Path};

pub fn save<P>(path: P, network: &Network) -> Result<()>
where
    P: AsRef<Path>,
{
    let mut file = File::create(path)?;
    bincode::serialize_into(&mut file, network)?;
    Ok(())
}

pub fn load<P>(path: P) -> Result<Network>
where
    P: AsRef<Path>,
{
    let mut file = File::open(path)?;
    let network = bincode::deserialize_from(&mut file)?;
    Ok(network)
}

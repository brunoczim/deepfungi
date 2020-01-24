pub trait Input {
    fn as_float(&self) -> f64;
}

impl Input for f64 {
    fn as_float(&self) -> f64 {
        *self
    }
}
